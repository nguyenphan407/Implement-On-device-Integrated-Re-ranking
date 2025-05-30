# producer_via_kafka.py

import json
import time
from kafka import KafkaProducer, KafkaAdminClient
from kafka.admin import NewTopic
import numpy as np
import tensorflow as tf
import pickle

from train_DIR import reranker_parse_args, load_data
from DIR import DIR

# Kafka config
BOOTSTRAP_SERVERS = ['localhost:9092']
TOPIC = 'user-tracking'

def create_topic(admin, topic_name):
    try:
        admin.create_topics([NewTopic(name=topic_name, num_partitions=1, replication_factor=1)])
        print(f"Created topic {topic_name}")
    except:
        print(f"Topic {topic_name} exists, skipping creation.")

def main():
    # 1) Parse args and load data via load_data to get 8 arrays
    args = reranker_parse_args()
    test_set = load_data(
        args.test_data_dir, args.cate_idx_map, args.cate_idx,
        args.list_len, args.union_hist_len, args.cate_hist_len,
        split_candi=True, split_hist=True, cloud_hist=True,
        reverse_hist=False, union_hist=True
    )
    N = len(test_set[0])
    print(f"Loaded {N} records via load_data()")

    # 2) Initialize producer
    admin = KafkaAdminClient(bootstrap_servers=BOOTSTRAP_SERVERS)
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    create_topic(admin, TOPIC)

    # 3) Loop each record, prepare feed data and send
    # Initialize a dummy model to use prepare_feed_data
    tf.compat.v1.disable_eager_execution()
    model = DIR(args, loss='STE')
    # We only need prepare_feed_data, no session
    for i in range(N):
        # Build single-sample batch_data tuple
        batch_data = tuple(arr[i:i+1] for arr in test_set)
        feed = model.prepare_feed_data(batch_data, split_hist=True, union_hist=True)
        # Add keep_prob and is_train
        feed[model.keep_prob] = 1.0
        feed[model.is_train] = False

        # Convert feed dict: placeholder.name -> list
        payload = {}
        for ph, arr in feed.items():
            # strip ":0" from name
            name = ph.name.split(':')[0]
            arr_val = arr.tolist() if isinstance(arr, np.ndarray) else arr
            payload[name] = arr_val

        producer.send(TOPIC, payload)
        print(f"Send Record {i}: {payload.keys()}")
        time.sleep(5)

    producer.flush()
    print("All records sent.")

if __name__ == '__main__':
    main()
