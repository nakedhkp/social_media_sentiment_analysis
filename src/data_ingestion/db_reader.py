import configparser
import os
from pyspark.sql import SparkSession

def get_db_properties():
    """从 config.ini 读取数据库连接属性"""
    config = configparser.ConfigParser()
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(base_dir, 'config', 'config.ini')

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件未找到: {config_path}。"
                                "请在 config/config.ini 中创建并配置你的数据库凭证。")
    config.read(config_path)
    
    if 'mysql_db' not in config:
        raise KeyError("配置文件 config.ini 中缺少 'mysql_db' 部分。")

    db_config = config['mysql_db']
    properties = {
        "user": db_config.get('user'),
        "password": db_config.get('password'),
        "driver": "com.mysql.jdbc.Driver" 
    }
    # 确保所有必要的配置项都存在
    for key in ['host', 'database', 'user', 'password']:
        if not db_config.get(key):
            raise ValueError(f"配置文件 config.ini 的 'mysql_db' 部分缺少必要项: '{key}'")
            
    url = f"jdbc:mysql://{db_config.get('host')}:{db_config.get('port', '3306')}/{db_config.get('database')}"
    return url, properties

def read_table_from_db(spark: SparkSession, table_name: str):
    """
    从MySQL数据库读取一个表到Spark DataFrame。
    """
    url, properties = get_db_properties()
    print(f"正在从数据库读取表 '{table_name}' ...")
    try:
        df = spark.read.jdbc(url=url, table=table_name, properties=properties)
        return df
    except Exception as e:
        print(f"从数据库读取表 {table_name} 时出错: {e}")
        if "No suitable driver found" in str(e) or "ClassNotFoundException" in str(e):
            print("错误: 未找到MySQL JDBC驱动。请确保 'mysql:mysql-connector-java' "
                  "已正确配置在 spark.jars.packages 或 spark.jars 中，并且版本正确。")
        raise