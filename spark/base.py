import abc
from pyspark.sql import SparkSession

class Inference(abc.ABC):
    def __init__(self, config: dict = None):
        self.spark = SparkSession \
            .builder \
            .master("local[*]") \
            .appName("Inference") \
            .config("spark.jars.packages",
                    "org.apache.kafka:kafka-clients:3.2.1,"
                    "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1,") \
            .getOrCreate()

        self.config = config or {}

    @staticmethod
    def process(self, df):
        # Placeholder for processing logic
        # This method should be overridden in subclasses
        return df

    @abc.abstractmethod
    def read(self):
        pass

    @abc.abstractmethod
    def write(self, df):
        pass

    @staticmethod
    def run(self):
        df = self.read()
        result_df = self.process(df)
        self.write(result_df)