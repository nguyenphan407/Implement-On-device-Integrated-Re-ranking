import tensorflow as tf
import numpy as np
import json
from kafka import KafkaConsumer

# Configuration
KAFKA_BOOTSTRAP_SERVERS = ['localhost:9092']
KAFKA_TOPIC = 'user-tracking'
SAVED_MODEL_DIR = './model/taobao/saved_model'

def load_saved_model(session, model_dir):
    """
    Load a TensorFlow SavedModel and return the session and signature def.
    """
    meta_graph_def = tf.compat.v1.saved_model.loader.load(
        session,
        [tf.compat.v1.saved_model.tag_constants.SERVING],
        model_dir
    )
    signature = meta_graph_def.signature_def[
        tf.compat.v1.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    ]
    return signature

def main():
    # Disable eager to use TF1-style graphs
    tf.compat.v1.disable_eager_execution()

    # 1. Create session and load SavedModel
    sess = tf.compat.v1.Session()
    signature = load_saved_model(sess, SAVED_MODEL_DIR)
    graph = sess.graph

    # 2. Extract input and output tensors from the signature def
    input_tensors = {
        name: graph.get_tensor_by_name(tensor_info.name)
        for name, tensor_info in signature.inputs.items()
    }
    output_tensor = graph.get_tensor_by_name(
        signature.outputs['predictions'].name
    )

    # 3. Initialize Kafka consumer
    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )

    # 4. Consume messages and infer
    for msg in consumer:
        rec = msg.value  # dict keys match signature.inputs keys

        print(type(rec))

        feed_dict = {}
        for name, tensor in input_tensors.items():
            if name == 'labels':
                name = 'lb'
            if name == 'user_features':
                name = 'usr_ph'
            if name == 'keep_prob':
                feed_dict[tensor] = 1.0
            elif name == 'is_train':
                feed_dict[tensor] = False
            else:
                arr = np.array(rec[f"inputs/{name}"])
                # Add batch dimension if missing
                if arr.ndim == tensor.shape.ndims - 1:
                    arr = arr[np.newaxis, ...]
                feed_dict[tensor] = arr

        # Run inference
        preds = sess.run(output_tensor, feed_dict=feed_dict)
        print("Inference result:", preds)

if __name__ == "__main__":
    main()