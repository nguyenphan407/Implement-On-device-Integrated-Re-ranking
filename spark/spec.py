import pyspark.sql.types as T

input_schema = T.StructType([
    T.StructField('Time', T.StringType(), True),
    T.StructField('V1', T.DoubleType(), True),
    T.StructField('V2', T.DoubleType(), True),
    T.StructField('V3', T.DoubleType(), True),
    T.StructField('V4', T.DoubleType(), True),
    T.StructField('V5', T.DoubleType(), True),
    T.StructField('V6', T.DoubleType(), True),
    T.StructField('V7', T.DoubleType(), True),
    T.StructField('V8', T.DoubleType(), True),
    T.StructField('V9', T.DoubleType(), True),
    T.StructField('V10', T.DoubleType(), True),
    T.StructField('V11', T.DoubleType(), True),
    T.StructField('V12', T.DoubleType(), True),
    T.StructField('V13', T.DoubleType(), True),
    T.StructField('V14', T.DoubleType(), True),
    T.StructField('V15', T.DoubleType(), True),
    T.StructField('V16', T.DoubleType(), True),
    T.StructField('V17', T.DoubleType(), True),
    T.StructField('V18', T.DoubleType(), True),
    T.StructField('V19', T.DoubleType(), True),
    T.StructField('V20', T.DoubleType(), True),
    T.StructField('V21', T.DoubleType(), True),
    T.StructField('V22', T.DoubleType(), True),
    T.StructField('V23', T.DoubleType(), True),
    T.StructField('V24', T.DoubleType(), True),
    T.StructField('V25', T.DoubleType(), True),
    T.StructField('V26', T.DoubleType(), True),
    T.StructField('V27', T.DoubleType(), True),
])

output_schema = T.StructType([
    T.StructField('Time', T.StringType(), True),
    T.StructField('predicted_prob', T.DoubleType(), True)
])