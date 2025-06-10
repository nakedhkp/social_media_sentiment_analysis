# test_snownlp_udf.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType
from src.processing.sentiment_analyzer import get_sentiment_score

spark = SparkSession.builder.appName("TestSnowNLP").getOrCreate()
data = [("这是一个好消息",), ("",), (None,)]
df = spark.createDataFrame(data, ["text"])
sentiment_score_udf = udf(get_sentiment_score, FloatType())
df_with_score = df.withColumn("sentiment_score", sentiment_score_udf("text"))
df_with_score.show()
spark.stop()