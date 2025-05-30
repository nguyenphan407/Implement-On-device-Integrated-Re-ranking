from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.sql.functions import from_json, col
import tensorflow as tf
import numpy as np
import pandas as pd
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SAVED_MODEL_DIR = './model/taobao/saved_model'

# Module‐level lazy loader and infer_batch
_input_tensors = None
_output_tensor = None
_sess = None


def _load_saved_model(session, model_dir):
    """
    Load a TensorFlow SavedModel and return the session and signature def.
    """
    logger.info(f'Loading SavedModel from {model_dir}')
    meta_graph_def = tf.compat.v1.saved_model.loader.load(
        session,
        [tf.compat.v1.saved_model.tag_constants.SERVING],
        model_dir
    )
    signature = meta_graph_def.signature_def[
        tf.compat.v1.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    ]
    logger.info(f'Signature Def for SavedModel')
    return signature


def _lazy_load_tf():
    global _input_tensors, _output_tensor, _sess

    if _sess is not None:
        return _input_tensors, _output_tensor, _sess

    logger.info('Loading TensorFlow SavedModel')
    # Disable eager to use TF1-style graphs
    tf.compat.v1.disable_eager_execution()

    # 1. Create session and load SavedModel
    _sess = tf.compat.v1.Session()
    signature = _load_saved_model(_sess, SAVED_MODEL_DIR)
    graph = _sess.graph

    # 2. Extract input and output tensors from the signature def
    _input_tensors = {
        name: graph.get_tensor_by_name(tensor_info.name)
        for name, tensor_info in signature.inputs.items()
    }
    _output_tensor = graph.get_tensor_by_name(
        signature.outputs['predictions'].name
    )

    logger.info('Loaded SavedModel and extracted tensors.')
    return _input_tensors, _output_tensor, _sess


def df_to_dict_collect(dataframe):
    logger.info("Converting DataFrame to dict...")

    rows = dataframe.collect()
    result = []

    for row in rows:
        # Chuyển Row thành dict
        row_dict = row.asDict()
        result.append(row_dict)

    logger.info(f'Converted DataFrame to list of dict with {len(result)} records.')
    return result


def infer_batch(batch_df, batch_id):
    try:
        print(f"---------------------Batch {batch_id}---------------------")

        # # Show batch content for debugging
        # batch_df.show(truncate=False)

        # Check if batch is empty
        if batch_df.count() == 0:
            logger.info(f"Batch {batch_id} is empty, skipping inference")
            return

        # Load TensorFlow model
        input_tensors, output_tensor, sess = _lazy_load_tf()
        logger.info("Starting inference on batch...")

        # Convert DataFrame to dict
        logger.info(f'Processing batch {batch_id} with records.')
        batch_dict_list = df_to_dict_collect(batch_df)

        if not batch_dict_list:
            logger.info(f"No data in batch {batch_id}")
            return

        value_bytes = batch_dict_list[0]
        value_bytes = value_bytes['value']
        json_string = value_bytes.decode('utf-8')
        parsed_data = json.loads(json_string)

        print("Batch data:", parsed_data)

        # Process each record in the batch
        results = []
        feed = {}

        # Prepare feed dict for TensorFlow
        for name, tensor in input_tensors.items():
            if name == 'labels':
                name = 'lb'
            if name == 'user_features':
                name = 'usr_ph'
            if name == 'keep_prob':
                feed[tensor] = 1.0
            elif name == 'is_train':
                feed[tensor] = False
            else:
                arr = np.array(parsed_data[f"inputs/{name}"])
                # Add batch dimension if missing
                if arr.ndim == tensor.shape.ndims - 1:
                    arr = arr[np.newaxis, ...]
                feed[tensor] = arr

        # Run inference
        if feed:
            pred = sess.run(output_tensor, feed_dict=feed)
            results.append(pred.tolist())
            logger.info(f'Inference result: {pred}')

        # Convert results to DataFrame and display
        if results:
            from pyspark.sql import SparkSession
            spark = SparkSession.getActiveSession()
            results_df = spark.createDataFrame(
                [(result,) for result in results],
                ["predictions"]
            )
            results_df.show(truncate=False)

            # If you want to save results, you can write to another Kafka topic or file
            # results_df.write.mode("append").parquet("path/to/results")

        print("--------------------------------------------------------")

    except Exception as e:
        logger.error(f"Error in batch {batch_id}: {str(e)}")
        import traceback
        traceback.print_exc()


class StreamInference:
    def __init__(self, config):
        self.config = config
        self.spark = SparkSession.builder \
            .appName("StreamingInference") \
            .master("local[*]") \
            .config("spark.jars.packages",
                    "org.apache.kafka:kafka-clients:3.2.1,"
                    "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1") \
            .getOrCreate()

        # output schema: usr_ph + pred (Array of Double)
        self.output_schema = T.StructType([
            T.StructField("pred", T.ArrayType(T.ArrayType(T.IntegerType())))
        ])

    def read(self):
        logger.info(f'Start Reading from Kafka topic: {self.config["source"]["kafka_topics"]}')
        return (
            self.spark.readStream
            .format("kafka")
            .option("subscribe", self.config["source"]["kafka_topics"])
            .options(**self.config["source"]["options"])
            .load()
            # .select(from_json(col("value").cast("string")).alias("j"))
            # .select("j.*")
        )

    def write(self, df):
        logger.info("Starting streaming inference...")
        print(f"---------------------DataFrame Schema---------------------")
        df.printSchema()
        print("--------------------------------------------------------")

        query = (
            df.writeStream
            .outputMode("append")
            .foreachBatch(infer_batch)
            .option("checkpointLocation", self.config["sink"]["checkpoint_location"])
            .start()
        )

        query.awaitTermination()

    def run(self):
        df = self.read()
        self.write(df)


if __name__ == "__main__":
    stream_config = {
        "source": {
            "kafka_topics": "user-tracking",
            "options": {
                "kafka.bootstrap.servers": "localhost:9092",
                "startingOffsets": "latest"
            }
        },
        "sink": {
            "checkpoint_location": "spark/tmp/checkpoints"
        }
    }
    StreamInference(stream_config).run()