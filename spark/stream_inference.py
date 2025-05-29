from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.dataframe import DataFrame
from spark.base import Inference
from spec import input_schema, output_schema

class StreamInference(Inference):
    def read(self) -> DataFrame:
        return self.spark \
            .readStream \
            .format("kafka") \
            .option("subscribe", self.config["source"]["kafka_topics"]) \
            .options(**self.config["source"]["kafka_options"]) \
            .load() \
            .select(F.from_json(F.col("value").cast("string"),
                                input_schema).alias("value")) \
            .select(F.col("value.*"))

    def write(self, df) -> None:
        df \
        .select("value.*") \
        .writeStream \
        .format("kafka") \
        .options(**self.config["sink"]["options"]) \
        .option("checkpointLocation", self.config["sink"]["checkpoint_location"]) \
        .start(self.config["sink"]["kafka_topics"]) \
        .awaitTermination()


if __name__ == "__main__":
    stream_config = {
        "source": {
            "kafka_topics": "user_tracking",
            "options": {
                "kafka.bootstrap.servers": "localhost:9092",
                "startingOffsets": "latest",
            }
        },
        "sink": {
            "kafka_topics": "prediction",
            "checkpoint_location": "",
            "options": {
                "kafka.bootstrap.servers": "localhost:9092",
            }
        },
    }
    inference = StreamInference(stream_config)
    inference.run()