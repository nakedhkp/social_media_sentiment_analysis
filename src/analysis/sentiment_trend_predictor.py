from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, avg, count, when, lit, expr, date_format, from_unixtime,
    lag, mean, stddev, dayofweek, month, year
)
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import DoubleType, DateType
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

class EnhancedSentimentTrendPredictor:
    
    def __init__(self, spark):
        self.spark = spark
        self.trained_model = None
        self.feature_columns = []
        
    def prepare_features(self, df):
        print("📊 准备特征...")
        
        # 数据清理
        df = df.filter(
            col("sentiment_score").isNotNull() & 
            col("create_time").isNotNull()
        )
        
        # 转换时间格式并聚合
        df = df.withColumn("date", date_format(from_unixtime(col("create_time")), "yyyy-MM-dd"))
        
        # 按日期聚合
        daily_sentiment = df.groupBy("date") \
            .agg(
                avg("sentiment_score").alias("avg_sentiment"),
                count("*").alias("post_count"),
                stddev("sentiment_score").alias("sentiment_std")
            ) \
            .orderBy("date")
        
        # 创建时间特征
        window_spec = Window.orderBy("date")
        df_features = daily_sentiment \
            .withColumn("sentiment_lag_1", lag("avg_sentiment", 1).over(window_spec)) \
            .withColumn("sentiment_lag_2", lag("avg_sentiment", 2).over(window_spec)) \
            .withColumn("sentiment_ma_7", mean("avg_sentiment").over(window_spec.rowsBetween(-6, 0))) \
            .withColumn("date_parsed", col("date").cast(DateType())) \
            .withColumn("day_of_week", dayofweek(col("date_parsed"))) \
            .withColumn("month", month(col("date_parsed"))) \
            .withColumn("year", year(col("date_parsed")))
        
        # 处理缺失值
        df_features = df_features.fillna(0.0)
        
        print(f"✅ 特征准备完成，共 {df_features.count()} 天的数据")
        return df_features
    
    def prepare_training_data(self, df):
        """准备训练数据"""
        print("📋 准备训练数据...")
        
        # 选择特征列
        exclude_cols = ['date', 'date_parsed', 'avg_sentiment']
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        # 创建特征向量
        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol="features_raw",
            handleInvalid="skip"
        )
        
        # 特征标准化
        scaler = StandardScaler(
            inputCol="features_raw",
            outputCol="features",
            withStd=True,
            withMean=True
        )
        
        # 准备标签
        df_features = assembler.transform(df)
        df_features = df_features.withColumn("label", col("avg_sentiment").cast(DoubleType()))
        
        # 应用标准化
        scaler_model = scaler.fit(df_features)
        df_scaled = scaler_model.transform(df_features)
        
        self.feature_columns = feature_cols
        self.scaler_model = scaler_model
        
        return df_scaled
    
    def train_model(self, training_data):
        """训练模型"""
        print("🤖 训练随机森林模型...")
        
        model = RandomForestRegressor(
            featuresCol="features",
            labelCol="label",
            numTrees=50,
            maxDepth=10
        )
        
        pipeline = Pipeline(stages=[model])
        trained_model = pipeline.fit(training_data)
        
        print("✅ 模型训练完成")
        return trained_model
    
    def predict_future(self, model, last_data, days_to_predict=30):
        """预测未来趋势"""
        print(f"🔮 预测未来{days_to_predict}天...")
        
        predictions = []
        current_data = last_data.collect()[0].asDict()
        
        for i in range(days_to_predict):
            # 计算预测日期
            last_date = datetime.strptime(current_data['date'], '%Y-%m-%d')
            pred_date = last_date + timedelta(days=i+1)
            
            # 更新时间特征
            current_data['date'] = pred_date.strftime('%Y-%m-%d')
            current_data['day_of_week'] = pred_date.weekday() + 1
            current_data['month'] = pred_date.month
            current_data['year'] = pred_date.year
            
            # 创建预测数据框
            pred_row = self.spark.createDataFrame([current_data])
            
            # 重新装配特征
            assembler = VectorAssembler(
                inputCols=self.feature_columns,
                outputCol="features_raw",
                handleInvalid="skip"
            )
            pred_features = assembler.transform(pred_row)
            pred_scaled = self.scaler_model.transform(pred_features)
            
            # 进行预测
            prediction = model.transform(pred_scaled)
            pred_value = prediction.select("prediction").collect()[0]["prediction"]
            
            # 存储预测结果
            predictions.append({
                'date': current_data['date'],
                'predicted_sentiment': float(pred_value)
            })
            
            # 更新滞后特征
            current_data['sentiment_lag_2'] = current_data['sentiment_lag_1']
            current_data['sentiment_lag_1'] = pred_value
            current_data['avg_sentiment'] = pred_value
        
        # 转换为DataFrame
        predictions_df = self.spark.createDataFrame(predictions)
        print("✅ 预测完成")
        
        return predictions_df
    
    def run_prediction_pipeline(self, sentiment_df, test_size=0.2, days_to_predict=30):
        """运行预测流程"""
        print("🚀 开始预测流程...")
        
        try:
            # 1. 准备特征
            feature_df = self.prepare_features(sentiment_df)
            
            # 2. 准备训练数据
            training_data = self.prepare_training_data(feature_df)
            
            # 3. 划分训练集和测试集
            total_rows = training_data.count()
            train_size = int(total_rows * (1 - test_size))
            train_data = training_data.limit(train_size)
            test_data = training_data.subtract(train_data)
            
            # 4. 训练模型
            model = self.train_model(train_data)
            self.trained_model = model
            
            # 5. 评估模型
            evaluator = RegressionEvaluator(
                labelCol="label",
                predictionCol="prediction",
                metricName="rmse"
            )
            rmse = evaluator.evaluate(model.transform(test_data))
            print(f"📊 模型RMSE: {rmse:.4f}")
            
            # 6. 获取最后一天的数据用于预测
            last_data = feature_df.orderBy(col("date").desc()).limit(1)
            
            # 7. 预测未来趋势
            future_predictions = self.predict_future(model, last_data, days_to_predict)
            
            print("🎉 预测流程完成!")
            
            return {
                "model": model,
                "evaluation": {"rmse": rmse},
                "future_predictions": future_predictions
            }
            
        except Exception as e:
            print(f"❌ 预测流程出错: {str(e)}")
            raise e