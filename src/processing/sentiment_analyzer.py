import re
import sys
from pyspark.sql.functions import udf, col, when
from pyspark.sql.types import FloatType, StringType
from snownlp import SnowNLP

# ========== 文本清洗 ==========
def clean_text(text):
    """基础文本清理"""
    if not text or not isinstance(text, str):
        return ""
    try:
        # 去除URL
        text = re.sub(r'http[s]?://[^\s]+', '', text)
        # 去除@用户名
        text = re.sub(r'@[\w]+', '', text)
        # 去除特殊符号与多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        print(f"[clean_text 错误] 输入: {text} 错误: {e}", file=sys.stderr)
        return ""

# ========== 情感得分 ==========
def get_sentiment_score(text):
    """获取情感分数 (0-1)，越接近1越积极"""
    if not text or text.strip() == "":
        return None
    try:
        cleaned_text = clean_text(text)
        if not cleaned_text or cleaned_text.strip() == "":
            return None
        s = SnowNLP(cleaned_text)
        score = s.sentiments
        if isinstance(score, float):
            return max(0.0, min(score, 1.0))  
        return None
    except Exception as e:
        print(f"[get_sentiment_score 错误] 文本: {text[:50]}... 错误: {e}", file=sys.stderr)
        return None

# ========== 情感分类 ==========
def get_sentiment_category(score, positive_threshold=0.6, negative_threshold=0.4):
    """根据分数确定情感类别"""
    if score is None:
        return "unknown"
    try:
        if score >= positive_threshold:
            return "positive"
        elif score <= negative_threshold:
            return "negative"
        else:
            return "neutral"
    except Exception:
        return "unknown"

# ========== 注册UDF ==========
# 连接 纯 Python 代码 和 Spark 分布式计算环境 的桥梁
sentiment_score_udf = udf(get_sentiment_score, FloatType())
sentiment_category_udf = udf(lambda score: get_sentiment_category(score), StringType())

# ========== 分析入口 ==========
def analyze_sentiment_batch(spark_df):
    """对包含 `text` 字段的 Spark DataFrame 批量执行情感分析"""
    
    df_with_score = spark_df.withColumn("sentiment_score", sentiment_score_udf(col("text")))

    df_with_category = df_with_score.withColumn(
        "sentiment_category",
        when(col("sentiment_score").isNull(), "unknown")
        .when(col("sentiment_score") >= 0.6, "positive")
        .when(col("sentiment_score") <= 0.4, "negative")
        .otherwise("neutral")
    )

    return df_with_category
