import os
import sys
import csv
import pandas as pd
from pyspark.sql.functions import (
    col, count, avg, expr, from_unixtime, date_format,
    split, explode, lower, trim, concat_ws, when # 确保 when 被导入
)
from pyspark.sql.types import TimestampType, StringType, LongType, IntegerType
from analysis.sentiment_trend_predictor import EnhancedSentimentTrendPredictor

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.spark_utils import get_spark_session, stop_spark_session
from src.processing.sentiment_analyzer import analyze_sentiment_batch
from src.data_ingestion.simulator import generate_simulated_data
from src.data_ingestion.db_reader import read_table_from_db

def run_sentiment_analysis():
    """运行完整的情感分析流程"""
    
    spark = None
    try:
        # 1. 初始化Spark
        print("🚀 初始化Spark环境...")
        spark = get_spark_session("SocialMediaSentimentAnalysis")
        
        # 2. 生成或加载数据
        data_file = "data/raw/simulated_posts.csv"
        if not os.path.exists(data_file):
            print("📊 生成模拟数据...")
            generate_simulated_data(5000, data_file)
        
        # 3. 加载数据
        print("📖 加载数据...")
        df = spark.read.csv(data_file, header=True, inferSchema=True)
        
        print(f"数据总量: {df.count()} 条")
        print("数据结构:")
        df.printSchema()
        print("数据样例:")
        df.show(5, truncate=False)
        
        # 4. 数据清理
        print("🧹 数据清理...")
        df_clean = df.filter(col("text").isNotNull() & (col("text") != ""))

        
        # 5. 情感分析
        print("🎭 执行情感分析...")
        df_sentiment = analyze_sentiment_batch(df_clean)
        
        # 缓存结果以提高后续操作性能
        df_sentiment.cache()
        
        # 6. 基础统计
        print("\n📈 情感分析结果统计:")
        sentiment_stats = df_sentiment.groupBy("sentiment_category").agg(
            count("*").alias("count"),
            avg("sentiment_score").alias("avg_score")
        ).orderBy("count", ascending=False)
        
        sentiment_stats.show()
        
        # 7. 平台统计
        print("\n📱 各平台情感分布:")
        platform_sentiment = df_sentiment.groupBy("platform", "sentiment_category").count().orderBy("platform", "count")
        platform_sentiment.show()
        
        # 8. 保存结果
        print("💾 保存分析结果...")
        
        # 确保输出目录存在
        output_dir = "data/processed"
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存完整结果为Parquet格式
        df_sentiment.write.mode("overwrite").parquet(f"{output_dir}/sentiment_results.parquet")
        
        # 保存统计结果为CSV
        sentiment_stats_pd = sentiment_stats.toPandas()
        sentiment_stats_pd.to_csv(
            f"{output_dir}/sentiment_distribution.csv",
            index=False,
            quoting=csv.QUOTE_NONNUMERIC,
            encoding='utf-8-sig'
        )
        
        platform_sentiment_pd = platform_sentiment.toPandas()
        platform_sentiment_pd.to_csv(
            f"{output_dir}/platform_sentiment.csv",
            index=False,
            quoting=csv.QUOTE_NONNUMERIC,
            encoding='utf-8-sig'
        )
        
        # 保存详细结果的样本为CSV（用于前端展示）
        sample_results = df_sentiment.select("text", "sentiment_score", "sentiment_category", "platform").limit(1000)
        sample_results_pd = sample_results.toPandas()
        sample_results_pd.to_csv(
            f"{output_dir}/sentiment_sample.csv",
            index=False,
            quoting=csv.QUOTE_NONNUMERIC,
            encoding='utf-8-sig'
        )
        
        print("✅ 分析完成！结果已保存到 data/processed/ 目录")
        print(f"- 完整结果: {output_dir}/sentiment_results.parquet")
        print(f"- 情感分布: {output_dir}/sentiment_distribution.csv")
        print(f"- 平台分析: {output_dir}/platform_sentiment.csv")
        print(f"- 样本数据: {output_dir}/sentiment_sample.csv")
        
        return True
        
    except Exception as e:
        print(f"❌ 分析过程中出错: {e}")
        import traceback # 确保 traceback 被导入
        traceback.print_exc()
        return False
    
    finally:
        if spark:
            stop_spark_session(spark)


def run_offline_sql_analysis():
    """
    对来自 MySQL 的笔记(title + desc)和评论(content)一起进行情感分析，
    并按 IP 属地、时间等维度统计；同时对笔记标签进行拆分统计，供前端词云使用。
    最后进行情感趋势预测。
    """
    spark = None
    try:
        print("🚀 初始化 Spark 环境 (数据库 SQL 分析)...")
        spark = get_spark_session("SocialMediaSentimentAnalysis_DBSQL")

        # ----------------------------
        # 1. 从 MySQL 读取笔记表和评论表
        # ----------------------------
        print("📖 正在从 MySQL 数据库读取 xhs_note 表和 xhs_note_comment 表...")
        df_notes_raw = read_table_from_db(spark, "xhs_note") # 重命名为 _raw 以示区分
        df_comments_raw = read_table_from_db(spark, "xhs_note_comment") # 重命名为 _raw

        output_db_dir = "data/processed/db_analysis"
        print(f"📁 确保输出目录存在: {output_db_dir}")
        os.makedirs(output_db_dir, exist_ok=True)
        # -----------------------------------------------------------------
        # 2. 数据类型转换和初步清理 (在过滤之前进行，确保过滤条件基于正确类型)
        # -----------------------------------------------------------------
        print("⚙️ 进行数据类型转换和初步清理...")

        # --- 处理 df_notes ---
        df_notes = df_notes_raw \
            .withColumn( # 处理 liked_count
                "liked_count_cleaned",
                when(trim(col("liked_count")) == "", "0") # 空字符串视为 "0"
                .when(col("liked_count").isNull(), "0")   # NULL 视为 "0"
                .otherwise(col("liked_count"))
            ) \
            .withColumn("liked_count", col("liked_count_cleaned").cast(LongType())) \
            .drop("liked_count_cleaned") \
            .withColumn( # 处理 comment_count
                "comment_count_cleaned",
                when(trim(col("comment_count")) == "", "0") # 空字符串视为 "0"
                .when(col("comment_count").isNull(), "0")   # NULL 视为 "0"
                .otherwise(col("comment_count"))
            ) \
            .withColumn("comment_count", col("comment_count_cleaned").cast(LongType())) \
            .drop("comment_count_cleaned")


        print("笔记表数据结构:")
        df_notes.printSchema()

        df_comments = df_comments_raw

        # ----------------------------
        # 2.b 数据质量检查与过滤 (现在基于已转换和初步清理的数据)
        # ----------------------------
        print("🧹 数据质量检查与过滤...")
        # 笔记：要求 note_id 非空，title 或 desc 至少有一个不为空
        df_notes = df_notes.filter(
            col("note_id").isNotNull() &
            (
                (col("title").isNotNull() & (trim(col("title")) != "")) | # 使用 trim 避免纯空格
                (col("desc").isNotNull() & (trim(col("desc")) != ""))   # 使用 trim
            ) &
            col("time").isNotNull()  # 确保发布时间戳存在
        )

        # 评论：要求 note_id、comment_id、content、create_time 均不为空
        df_comments = df_comments.filter(
            col("note_id").isNotNull() &
            col("comment_id").isNotNull() &
            (col("content").isNotNull() & (trim(col("content")) != "")) & # 使用 trim
            col("create_time").isNotNull()
        )

        notes_count = df_notes.count()
        comments_count = df_comments.count()
        print(f"有效笔记数据: {notes_count} 条, 有效评论数据: {comments_count} 条。")
        if notes_count == 0 and comments_count == 0:
            print("❗❗警告: 笔记和评论数据都为空，退出分析。")
            return False

        # ----------------------------
        # 3. 统计每篇笔记的点赞数与评论数
        # ----------------------------
        df_notes.createOrReplaceTempView("notes") # notes 视图现在包含转换后的数值列
        df_comments.createOrReplaceTempView("comments")

        note_stats_query = """
            SELECT
                n.note_id,
                n.title,
                COALESCE(n.liked_count, 0) AS liked_count, 
                COALESCE(n.comment_count, 0) AS note_declared_comment_count,
                COUNT(c.comment_id) AS actual_comment_count
            FROM notes n
            LEFT JOIN comments c ON n.note_id = c.note_id
            GROUP BY n.note_id, n.title, n.liked_count, n.comment_count
            ORDER BY actual_comment_count DESC
        """
        df_note_stats = spark.sql(note_stats_query)
        print("\n📊 每篇笔记统计信息 (笔记表声明评论数 vs 实际评论表统计):")
        df_note_stats.show(10, truncate=False)

        # ----------------------------
        # 4. 评论按天分布统计
        # ----------------------------
        df_comments_with_date = df_comments.withColumn(
            "comment_date",
            date_format(from_unixtime((col("create_time") / 1000).cast("long")), "yyyy-MM-dd")
        )
        df_comment_daily_dist = (
            df_comments_with_date
            .groupBy("comment_date")
            .count()
            .withColumnRenamed("count", "num_comments")
            .orderBy("comment_date")
        )
        print("\n🗓️ 评论按天分布统计:")
        df_comment_daily_dist.show(10)

        # ----------------------------
        # 5. 合并笔记和评论进行情感分析
        # ----------------------------
        # 5.1 处理评论：重命名 content -> text，保留 create_time、ip_location、note_id
        df_comments_for_sentiment = (
            df_comments
            .select(
                col("comment_id").alias("id"),
                col("note_id"),
                col("content").alias("text"),
                (col("create_time") / 1000).cast("long").alias("create_time"), 
                col("ip_location")
            )
        )
        # 5.2 处理笔记：将 title + desc 拼接为 text，保留 time 作为 create_time，ip_location、note_id
        df_notes_for_sentiment = (
            df_notes
            .withColumn(
                "text",
                concat_ws(" ", col("title"), col("desc")) # title/desc 可能为 NULL，concat_ws 会处理
            )
            .select(
                col("note_id").alias("id"), # 作为唯一标识
                col("note_id").alias("source_note_id"), # 保留原始笔记ID用于关联或追溯
                col("text"),
                (col("time") / 1000).cast("long").alias("create_time"), 
                col("ip_location")
            )
        )

        df_notes_for_sentiment = df_notes_for_sentiment.filter(trim(col("text")) != "")
        
        # 确保列名和类型一致以便 unionByName
        # df_comments_for_sentiment 已经有 id, note_id, text, create_time, ip_location
        # df_notes_for_sentiment 有 id (来自note_id), source_note_id, text, create_time, ip_location
        # 为了 unionByName, 我们需要统一列。
        # 方案：给笔记的 id 也用 note_id, 然后添加一个 type 列区分是笔记还是评论。
        # 或者，如果不需要区分，且情感分析的 id 只是一个唯一键，当前 note_id as id 也可以。
        # 这里我们假设情感分析只需要一个唯一 id, text, create_time, ip_location
        # 笔记的 note_id 字段在 df_notes_for_sentiment 中已经通过 alias("id") 映射为 id
        # 评论的 note_id 字段在 df_comments_for_sentiment 中是 note_id.
        # 为了统一，我们确保两个 DataFrame 都有 `note_id` 字段供情感分析后可能的回溯，
        # 以及一个统一的 `id` 字段。

        df_comments_for_sentiment = df_comments_for_sentiment.withColumn("type", expr("'comment'"))
        df_notes_for_sentiment = df_notes_for_sentiment.withColumn("type", expr("'note'")) \
                                                       .withColumn("note_id", col("source_note_id")) \
                                                       .drop("source_note_id")


        print("Schema for comments before union:")
        df_comments_for_sentiment.printSchema()
        print("Schema for notes before union:")
        df_notes_for_sentiment.printSchema()


        # df_combined_texts = df_comments_for_sentiment.unionByName(df_notes_for_sentiment, allowMissingColumns=True)
        # 由于列名已尽量对齐，可以直接 unionByName
        # 确保选取的列是共有的，或者使用 allowMissingColumns=True (Spark 3.0+)
        # 为简单起见，我们只选取核心列进行情感分析
        common_columns = ["id", "text", "create_time", "ip_location", "note_id", "type"]
        df_combined_texts = df_comments_for_sentiment.select(common_columns).unionByName(df_notes_for_sentiment.select(common_columns))


        print(f"\n🎭 对评论和笔记（共 {df_combined_texts.count()} 条）一起执行情感分析...")
        if df_combined_texts.count() == 0:
            print("❗❗警告: 合并后用于情感分析的数据为空，跳过情感分析及后续步骤。")
        else:
            df_combined_sentiment = analyze_sentiment_batch(df_combined_texts)
            df_combined_sentiment.cache()
            print("合并后情感分析结果示例 (显示 note_id 和 type):")
            df_combined_sentiment.select("id", "note_id", "type", "text", "sentiment_score", "sentiment_category").show(5, truncate=False)

            # 5.3 合并后的整体情感分布统计
            print("\n📈 合并后（笔记 + 评论）情感分布统计:")
            combined_sentiment_summary = (
                df_combined_sentiment
                .groupBy("sentiment_category")
                .agg(
                    count("*").alias("count"),
                    avg("sentiment_score").alias("avg_score")
                )
                .orderBy(col("count").desc()) # 明确指定 col
            )
            combined_sentiment_summary.show()

            # 5.4 合并后 情感随时间变化趋势
            # create_time 已经是秒级时间戳
            df_combined_sentiment_dated = df_combined_sentiment.withColumn(
                "date",
                date_format(from_unixtime(col("create_time")), "yyyy-MM-dd")
            )
            sentiment_over_time = (
                df_combined_sentiment_dated
                .groupBy("date", "sentiment_category")
                .count()
                .orderBy("date", "sentiment_category")
            )
            print("\n📉 合并后情感随时间变化趋势:")
            sentiment_over_time.show(10)

            # 5.5 合并后 按 IP 属地统计情感分布
            sentiment_by_location = (
                df_combined_sentiment
                .filter(col("ip_location").isNotNull() & (trim(col("ip_location")) != "")) # 使用 trim
                .groupBy("ip_location", "sentiment_category")
                .count()
                .orderBy(col("count").desc())
            )
            print("\n🗺️ 合并后评论+笔记情感按 IP 属地分布 (Top 10):")
            sentiment_by_location.show(10, truncate=False)
            
            # --- 保存情感分析相关结果 ---
            output_db_dir = "data/processed/db_analysis" 
            
            combined_sentiment_summary.toPandas().to_csv(
                f"{output_db_dir}/combined_sentiment_distribution.csv",
                index=False,
                quoting=csv.QUOTE_NONNUMERIC,
                encoding='utf-8-sig'
            )
            sentiment_over_time.toPandas().to_csv(
                f"{output_db_dir}/combined_sentiment_over_time.csv",
                index=False,
                quoting=csv.QUOTE_NONNUMERIC,
                encoding='utf-8-sig'
            )
            sentiment_by_location.limit(200).toPandas().to_csv(
                f"{output_db_dir}/combined_sentiment_by_location_sample.csv",
                index=False,
                quoting=csv.QUOTE_NONNUMERIC,
                encoding='utf-8-sig'
            )
            # 可以考虑保存 df_combined_sentiment 的样本或全量 (Parquet)
            df_combined_sentiment.select("id", "note_id", "type", "text", "sentiment_score", "sentiment_category", "create_time", "ip_location") \
                .limit(5000) \
                .toPandas() \
                .to_csv(
                    f"{output_db_dir}/combined_sentiment_sample_details.csv",
                    index=False,
                    quoting=csv.QUOTE_NONNUMERIC,
                    encoding='utf-8-sig',
                    escapechar='\\'  # 添加转义字符
                )

            # ----------------------------
            # 5.6 情感趋势预测
            # ----------------------------
            print("\n🔮 开始情感趋势预测...")
            predictor = EnhancedSentimentTrendPredictor(spark)
            
            # 准备预测数据
            prediction_data = df_combined_sentiment.select(
                "sentiment_score",
                "create_time"
            )
            
            # 运行预测流程
            prediction_results = predictor.run_prediction_pipeline(
                prediction_data,
                test_size=0.2,
                days_to_predict=30
            )
            
            # 保存预测结果
            future_predictions = prediction_results["future_predictions"]
            future_predictions.toPandas().to_csv(
                f"{output_db_dir}/sentiment_trend_predictions.csv",
                index=False,
                quoting=csv.QUOTE_NONNUMERIC,
                encoding='utf-8-sig'
            )
            
            # 保存模型评估结果
            evaluation_metrics = prediction_results["evaluation"]
            pd.DataFrame([evaluation_metrics]).to_csv(
                f"{output_db_dir}/sentiment_trend_evaluation.csv",
                index=False,
                quoting=csv.QUOTE_NONNUMERIC,
                encoding='utf-8-sig'
            )
            
            print("\n📈 情感趋势预测结果:")
            future_predictions.show(10)
            print("\n📊 模型评估指标:")
            print(f"- RMSE: {evaluation_metrics['rmse']:.4f}")

        # ----------------------------
        # 6. 热门标签分析（tag_list 以逗号分隔）
        # ----------------------------
        print("\n🔥 热门标签分析 (从笔记 tag_list 拆分):")
        df_tags = (
            df_notes # 使用已经过类型转换和初步过滤的 df_notes
            .select("note_id", "tag_list")
            .filter(col("tag_list").isNotNull() & (trim(col("tag_list")) != "")) # 使用 trim
            .withColumn("tag", explode(split(trim(col("tag_list")), ","))) # 对 tag_list 也 trim 一下
            .select(
                lower(trim(col("tag"))).alias("tag") # 对单个 tag 也 trim
            )
            .filter(trim(col("tag")) != "") # 确保 tag 非空
            .groupBy("tag")
            .agg(count("*").alias("tag_count"))
            .orderBy(col("tag_count").desc())
        )
        df_tags.show(10, truncate=False)

        # ----------------------------
        # 7. 保存所有分析结果到本地
        # ----------------------------
        output_db_dir = "data/processed/db_analysis" # 目录变量复用
        os.makedirs(output_db_dir, exist_ok=True) # 再次确保，虽然前面可能有

        print(f"\n💾 正在保存数据库分析结果到 {output_db_dir}/ ...")
        # 7.1 保存笔记点赞/评论统计
        df_note_stats.toPandas().to_csv(
            f"{output_db_dir}/note_comment_stats.csv",
            index=False,
            quoting=csv.QUOTE_NONNUMERIC,
            encoding='utf-8-sig'
        )
        # 7.2 保存评论按天分布
        df_comment_daily_dist.toPandas().to_csv(
            f"{output_db_dir}/comment_daily_distribution.csv",
            index=False,
            quoting=csv.QUOTE_NONNUMERIC,
            encoding='utf-8-sig'
        )
        # 7.6 保存热门标签，用于前端词云
        df_tags.limit(200).toPandas().to_csv(
            f"{output_db_dir}/top_tags_for_wordcloud.csv",
            index=False,
            quoting=csv.QUOTE_NONNUMERIC,
            encoding='utf-8-sig'
        )

        print(f"✅ 数据库 SQL 分析完成！结果已保存至 {output_db_dir}/ 目录")
        return True

    except Exception as e:
        print(f"❌ 数据库 SQL 分析过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        if spark:
            stop_spark_session(spark)

if __name__ == "__main__":
    print("=== Step 1: 运行模拟数据情感分析流程 ===")
    run_first = True # 控制是否运行模拟数据分析，方便调试
    if run_first:
        success1 = run_sentiment_analysis()
        if not success1:
            print("模拟数据分析失败，请检查。")
    else:
        success1 = True # 假设成功以便继续
        print("跳过模拟数据分析流程。")


    print("\n=== Step 2: 运行数据库离线 SQL 分析流程 ===")
    success2 = run_offline_sql_analysis()

    if success1 and success2: # 确保两个都定义了
        print("\n🎉 所有选定流程执行成功！现在可以运行前端查看结果：")
        print("👉 streamlit run frontend/app.py")
    else:
        print("\n⚠️ 有部分流程执行失败或未执行，请检查日志信息。")