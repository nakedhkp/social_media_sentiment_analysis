from pyspark.sql import SparkSession

def get_spark_session(app_name):
    """创建并返回一个SparkSession实例"""
    builder = SparkSession.builder \
        .appName(app_name) \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.network.timeout", "600s") \
        .config("spark.executor.heartbeatInterval", "60s") \
        .config("spark.driver.maxResultSize", "2g") \
        .config("spark.sql.shuffle.partitions", "10") \
        .config("spark.local.dir", "temp") \
        .config("spark.python.worker.memory", "512m") \
        .config("spark.jars.packages", "mysql:mysql-connector-java:8.0.28")

    spark = builder.getOrCreate()
    return spark

def stop_spark_session(spark):
    """停止SparkSession并清理临时文件"""
    try:
        spark.stop()
        # 清理临时目录
        import shutil
        import os
        temp_dir = "temp"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception as e:
        print(f"停止Spark会话时出错: {e}")
