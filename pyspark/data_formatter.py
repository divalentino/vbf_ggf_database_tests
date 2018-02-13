from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col

# Format the data in a way that can be read
# by decision trees.
def data_formatter(input_data, data_type, variables) :

    assembler = VectorAssembler(
        inputCols=variables,
        outputCol="features")

    transformed = assembler.transform(input_data);
    vec_var = transformed.select(col("weight"),col("features"))\
      .rdd\
      .map(lambda row: (data_type, row.weight, row.features)).toDF()\
      .select(col("_1").alias("label"),col("_2").alias("weight"),col("_3").alias("features"));

    return vec_var
