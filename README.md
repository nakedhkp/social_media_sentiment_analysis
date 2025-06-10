    ### 如何使用本系统：

    1. **环境准备：**
       - 确保已安装并启动 MySQL 数据库。
       - 使用项目提供的 SQL 脚本创建 `xhs_note` 和 `xhs_note_comment` 表，并导入测试数据。
       - 在 `config/config.ini` 中配置好你的 MySQL 连接信息。
       - 请将中文字体文件（如 `simhei.ttf`、`msyh.ttc`）放到 `frontend/assets/` 目录，并确保 `FONT_PATH` 指向该文件，以支持中文显示。

    2. **安装依赖：**
       ```bash
       pip install -r requirements.txt
       ```

    3. **运行分析脚本 (`src/main.py`)：**
       - **模拟数据分析：**
         ```bash
         python src/main.py simulation
         ```
       - **数据库(小红书)分析 (包含情感趋势预测)：**
         ```bash
         python src/main.py db_sql
         ```
       分析完成后，会在 `data/processed/`（模拟数据）或
       `data/processed/db_analysis/`（数据库）下生成对应 CSV 文件，包括情感趋势预测和评估结果。

    4. **启动前端应用：**
       ```bash
       streamlit run frontend/app.py
       ```
       打开浏览器，在侧边栏选择"模拟数据分析"或"数据库(小红书)分析"查看结果。

    ### 功能亮点：
    - 使用 PySpark 进行海量数据清洗与聚合。
    - 基于 SnowNLP 对中文文本做情感打分与分类。
    - **新增：** 使用 `EnhancedSentimentTrendPredictor` 模块对情感趋势进行预测，并展示预测结果与模型评估指标。
    - 可视化整体情感分布、按日趋势、热门标签、地理情感分布等。
    - 支持中文词云展示热门笔记标签。
