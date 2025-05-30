from spark.base import Inference

class BatchInference(Inference):
    def read(self):
        return self.spark \
            .read \
            .format("csv") \
            .load("dataset_path", header=True, inferSchema=True)

    def write(self):
        return self.spark \
            .write \
            .format("parquet") \
            .mode("overwrite") \
            .save("output_path")

if __name__ == "__main__":
    inference = BatchInference()
    inference.run()