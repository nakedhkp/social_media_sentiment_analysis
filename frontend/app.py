import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os
from wordcloud import WordCloud

# --- 页面配置 ---
st.set_page_config(
    page_title="社交媒体舆情分析系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 中文字体设置 ---
FONT_DIR = os.path.join(os.path.dirname(__file__), 'assets')
FONT_FILENAME = 'simhei.ttf'  
FONT_PATH = os.path.join(FONT_DIR, FONT_FILENAME)

if not os.path.exists(FONT_PATH):
    st.warning(f"字体文件 {FONT_PATH} 未找到，中文可能无法正常显示。")
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft YaHei', 'SimHei', 'DejaVu Sans']
else:
    plt.rcParams['font.sans-serif'] = [
        os.path.splitext(FONT_FILENAME)[0],
        'Arial Unicode MS', 'Microsoft YaHei', 'SimHei', 'DejaVu Sans'
    ]
plt.rcParams['axes.unicode_minus'] = False

# --- 数据加载函数 ---
@st.cache_data(ttl=300)
def load_simulated_data():
    try:
        sim_sentiment_dist = pd.read_csv("data/processed/sentiment_distribution.csv")
        sim_platform_sentiment = pd.read_csv("data/processed/platform_sentiment.csv")
        sim_sample_data = pd.read_csv("data/processed/sentiment_sample.csv")
        return sim_sentiment_dist, sim_platform_sentiment, sim_sample_data
    except FileNotFoundError:
        st.error("❌ 模拟数据分析结果文件未找到！请先运行 `python src/main.py simulation`。")
        return None, None, None
    except Exception as e:
        st.error(f"❌ 加载模拟数据时出错: {e}")
        return None, None, None

@st.cache_data(ttl=300)
def load_db_analysis_data():    
    base_path = "data/processed/db_analysis"
    data_files = {
        "note_stats": "note_comment_stats.csv",
        "comment_daily_dist": "comment_daily_distribution.csv",
        "combined_sentiment_dist": "combined_sentiment_distribution.csv",
        "sentiment_over_time": "combined_sentiment_over_time.csv", # 历史情感数据
        "top_tags": "top_tags_for_wordcloud.csv",
        "sentiment_by_location": "combined_sentiment_by_location_sample.csv",
        "trend_predictions": "sentiment_trend_predictions.csv", # 新增：情感趋势预测结果
        "trend_evaluation": "sentiment_trend_evaluation.csv"   # 新增：趋势模型评估结果
    }
    loaded_data = {}
    any_loaded = False

    for key, filename in data_files.items():
        file_path = os.path.join(base_path, filename)
        try:
            loaded_data[key] = pd.read_csv(file_path)
            any_loaded = True
        except FileNotFoundError:
            st.warning(f"文件未找到：{file_path}。对应图表或数据可能无法显示。")
            loaded_data[key] = pd.DataFrame() # 返回空DataFrame以便后续检查
        except Exception as e:
            st.error(f"❌ 加载 {filename} 时出错: {e}")
            loaded_data[key] = pd.DataFrame()

    if not any_loaded:
        st.error("❌ 数据库分析结果文件均未找到！请先运行 `python src/main.py db_sql`。")
        return None

    return loaded_data

# --- 主应用逻辑 ---
st.title("📊 社交媒体舆情分析系统")
st.markdown("基于 PySpark、SnowNLP 和 Streamlit 的舆情监控与分析平台")

# --- 侧边栏 ---
st.sidebar.header("⚙️ 控制面板")
analysis_mode = st.sidebar.radio(
    "选择分析模式：",
    ('模拟数据分析', '数据库(小红书)分析'),
    key="analysis_mode_selector"
)

# --- 颜色映射 ---
SENTIMENT_COLORS = {
    'positive': '#2E8B57',  # SeaGreen
    'negative': '#DC143C',  # Crimson
    'neutral': '#4682B4',   # SteelBlue
    'unknown': '#808080'    # Gray
}

if analysis_mode == '模拟数据分析':
    st.header("🚀 模拟数据分析结果")
    sentiment_dist, platform_sentiment, sample_data = load_simulated_data()

    if sentiment_dist is not None and not sentiment_dist.empty:
        # 指标卡
        col1, col2, col3, col4 = st.columns(4)
        total_posts = int(sentiment_dist['count'].sum())
        positive_posts = int(sentiment_dist.loc[sentiment_dist['sentiment_category'] == 'positive', 'count'].sum())
        negative_posts = int(sentiment_dist.loc[sentiment_dist['sentiment_category'] == 'negative', 'count'].sum())
        neutral_posts = int(sentiment_dist.loc[sentiment_dist['sentiment_category'] == 'neutral', 'count'].sum())

        col1.metric("📝 总帖子数", f"{total_posts:,}")
        col2.metric("😊 积极情感", f"{positive_posts:,}", f"{positive_posts/total_posts*100:.1f}%" if total_posts > 0 else "0.0%")
        col3.metric("😢 消极情感", f"{negative_posts:,}", f"{negative_posts/total_posts*100:.1f}%" if total_posts > 0 else "0.0%")
        col4.metric("😐 中性情感", f"{neutral_posts:,}", f"{neutral_posts/total_posts*100:.1f}%" if total_posts > 0 else "0.0%")

        # 情感分布图和条形图
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.subheader("🎭 整体情感分布 (模拟数据)")
            fig_pie = px.pie(
                sentiment_dist,
                values='count',
                names='sentiment_category',
                color='sentiment_category',
                color_discrete_map=SENTIMENT_COLORS,
                title="情感类别占比"
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)

        with chart_col2:
            st.subheader("📊 情感类别统计 (模拟数据)")
            fig_bar = px.bar(
                sentiment_dist.sort_values('count', ascending=True),
                x='count',
                y='sentiment_category',
                orientation='h',
                color='sentiment_category',
                color_discrete_map=SENTIMENT_COLORS,
                title="各类别帖子数量"
            )
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

        # 各平台情感热力图
        if platform_sentiment is not None and not platform_sentiment.empty:
            st.subheader("📱 各平台情感分析 (模拟数据)")
            if {'platform', 'sentiment_category', 'count'}.issubset(platform_sentiment.columns):
                try:
                    platform_pivot = platform_sentiment.pivot_table(
                        index='platform',
                        columns='sentiment_category',
                        values='count',
                        fill_value=0
                    )
                    fig_heatmap = px.imshow(
                        platform_pivot,
                        color_continuous_scale='RdYlGn',
                        title="各平台情感分布热力图"
                    )
                    fig_heatmap.update_xaxes(title="情感类别")
                    fig_heatmap.update_yaxes(title="平台")
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                except Exception as e:
                    st.error(f"生成平台情感热力图时出错: {e}")
                    st.dataframe(platform_sentiment)
            else:
                st.info("平台情感数据列不完整，无法生成热力图。")

        # 详细样本展示
        if sample_data is not None and not sample_data.empty:
            st.subheader("📋 详细分析结果 (模拟数据样本)")
            sentiments_sim = ["全部"] + list(sample_data['sentiment_category'].unique())
            sentiment_filter_sim = st.selectbox("选择情感类别 (模拟)", sentiments_sim, key="sim_sentiment_filter")

            platform_exists_sim = 'platform' in sample_data.columns
            platforms_sim = ["全部"] + list(sample_data['platform'].unique()) if platform_exists_sim else ["全部"]
            platform_filter_sim = st.selectbox(
                "选择平台 (模拟)", platforms_sim, key="sim_platform_filter",
                disabled=not platform_exists_sim
            )

            filtered_sim = sample_data.copy()
            if sentiment_filter_sim != "全部":
                filtered_sim = filtered_sim[filtered_sim['sentiment_category'] == sentiment_filter_sim]
            if platform_filter_sim != "全部" and platform_exists_sim:
                filtered_sim = filtered_sim[filtered_sim['platform'] == platform_filter_sim]

            st.write(f"共 {len(filtered_sim)} 条记录")
            display_cols_sim = ['text', 'sentiment_score', 'sentiment_category']
            if platform_exists_sim:
                display_cols_sim.append('platform')
            st.dataframe(filtered_sim[display_cols_sim], height=300, use_container_width=True)
        else:
            st.info("模拟数据样本为空或加载失败。")
    else:
        st.info("模拟数据分析结果为空或加载失败。")

elif analysis_mode == '数据库(小红书)分析':
    st.header("💾 数据库 (小红书) 分析结果")
    db_data = load_db_analysis_data()

    if db_data:
        # --- 评论情感分布 ---
        st.subheader("💬 笔记+评论 总体情感分布")
        combined_sentiment_dist = db_data.get("combined_sentiment_dist")
        if combined_sentiment_dist is not None and not combined_sentiment_dist.empty:
            fig_pie_db = px.pie(
                combined_sentiment_dist,
                values='count',
                names='sentiment_category',
                color='sentiment_category',
                color_discrete_map=SENTIMENT_COLORS,
                title="笔记+评论 情感类别占比"
            )
            fig_pie_db.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie_db, use_container_width=True)
        else:
            st.info("情感分布数据为空或加载失败。")

        # --- 情感趋势变化 (历史数据) ---
        st.subheader("📈 历史情感趋势变化 (按日，笔记+评论)")
        sentiment_over_time_hist = db_data.get("sentiment_over_time")
        if sentiment_over_time_hist is not None and not sentiment_over_time_hist.empty and 'date' in sentiment_over_time_hist.columns:
            sentiment_over_time_hist['date'] = pd.to_datetime(sentiment_over_time_hist['date'])
            sentiment_over_time_hist = sentiment_over_time_hist.sort_values('date')
            try:
                hist_trend_pivot = sentiment_over_time_hist.pivot_table(
                    index='date',
                    columns='sentiment_category',
                    values='count',
                    fill_value=0
                ).reset_index()
                fig_hist_trend = go.Figure()
                for sentiment_cat in SENTIMENT_COLORS.keys():
                    if sentiment_cat in hist_trend_pivot.columns:
                        fig_hist_trend.add_trace(go.Scatter(
                            x=hist_trend_pivot['date'],
                            y=hist_trend_pivot[sentiment_cat],
                            mode='lines+markers',
                            name=f"历史 {sentiment_cat}",
                            marker_color=SENTIMENT_COLORS[sentiment_cat]
                        ))
                fig_hist_trend.update_layout(
                    title='每日情感数量历史变化 (笔记 + 评论)',
                    xaxis_title='日期',
                    yaxis_title='数量',
                    legend_title_text='情感类别'
                )
                st.plotly_chart(fig_hist_trend, use_container_width=True)
            except Exception as e:
                st.error(f"绘制历史情感趋势图时出错: {e}")
                st.dataframe(sentiment_over_time_hist)
        else:
            st.info("历史趋势变化数据为空、加载失败或缺少 'date' 列。")

        # --- 情感趋势预测 ---
        st.subheader("🔮 未来情感趋势预测")
        trend_predictions = db_data.get("trend_predictions")
        trend_evaluation = db_data.get("trend_evaluation")

        if trend_predictions is not None and not trend_predictions.empty and 'date' in trend_predictions.columns and 'predicted_sentiment' in trend_predictions.columns:
            trend_predictions['date'] = pd.to_datetime(trend_predictions['date'])
            trend_predictions = trend_predictions.sort_values('date')

            # 创建预测趋势图
            fig_pred_trend = go.Figure()
            
            # 添加预测线
            fig_pred_trend.add_trace(go.Scatter(
                x=trend_predictions['date'],
                y=trend_predictions['predicted_sentiment'],
                mode='lines+markers',
                name='预测情感值',
                line=dict(color='royalblue', dash='dash'),
                marker=dict(size=8)
            ))
            
            # 如果有其他预测指标，也添加到图中
            if 'post_count' in trend_predictions.columns:
                fig_pred_trend.add_trace(go.Scatter(
                    x=trend_predictions['date'],
                    y=trend_predictions['post_count'],
                    mode='lines',
                    name='预测帖子数',
                    line=dict(color='green', dash='dot'),
                    yaxis='y2'
                ))
            
            # 更新布局
            fig_pred_trend.update_layout(
                title='未来情感趋势预测',
                xaxis_title='预测日期',
                yaxis_title='预测情感平均分',
                yaxis2=dict(
                    title='预测帖子数',
                    overlaying='y',
                    side='right'
                ),
                legend_title_text='预测指标',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_pred_trend, use_container_width=True)

            # --- 预测模型评估指标 ---
            if trend_evaluation is not None and not trend_evaluation.empty:
                st.markdown("#### 📈 预测模型评估指标 (测试集)")
                eval_metrics = trend_evaluation.iloc[0].to_dict()
                
                # 创建评估指标卡片
                cols_eval = st.columns(len(eval_metrics))
                for i, (metric, value) in enumerate(eval_metrics.items()):
                    if isinstance(value, float):
                        # 根据指标类型设置不同的颜色
                        if metric.lower() in ['rmse', 'mae']:
                            delta_color = 'inverse'  # 越小越好
                        else:
                            delta_color = 'normal'   # 越大越好
                            
                        cols_eval[i].metric(
                            label=metric.upper(),
                            value=f"{value:.4f}",
                            delta_color=delta_color
                        )
                    else:
                        cols_eval[i].metric(
                            label=metric.upper(),
                            value=str(value)
                        )
                
                # 添加预测结果说明
                st.markdown("""
                **预测结果说明：**
                - RMSE (均方根误差): 预测值与实际值的平均偏差，越小越好
                - MAE (平均绝对误差): 预测值与实际值的平均绝对偏差，越小越好
                - R² (决定系数): 模型解释的方差比例，越接近1越好
                """)
            else:
                st.info("预测模型评估数据为空或加载失败。")
        else:
            st.info("情感趋势预测数据为空、加载失败或列名不正确 (需要 'date' 和 'predicted_sentiment')。")

        # --- 评论内容词云图（改为标签词云） ---
        st.subheader("☁️ 热门标签词云图")
        top_tags = db_data.get("top_tags")
        if top_tags is not None and not top_tags.empty and {'tag', 'tag_count'}.issubset(top_tags.columns):
            # 构建词频字典
            tag_freq = dict(zip(top_tags['tag'], top_tags['tag_count']))
            if tag_freq:
                try:
                    if not os.path.exists(FONT_PATH):
                        st.error(f"词云字体 {FONT_PATH} 未找到，无法生成词云图。")
                    else:
                        wc = WordCloud(
                            font_path=FONT_PATH,
                            width=800,
                            height=400,
                            background_color='white',
                            stopwords=set(),  # 标签通常不需要停用词过滤
                            collocations=False
                        )
                        wordcloud_image = wc.generate_from_frequencies(tag_freq)
                        fig_wc, ax_wc = plt.subplots(figsize=(12, 6))
                        ax_wc.imshow(wordcloud_image, interpolation='bilinear')
                        ax_wc.axis('off')
                        st.pyplot(fig_wc)
                except Exception as e:
                    st.error(f"生成标签词云图时出错: {e}")
            else:
                st.info("热门标签数据为空，无法生成词云图。")
        else:
            st.info("热门标签数据缺失或列名不正确。")

        # --- 热门笔记标签柱状图 ---
        st.subheader("🏷️ 热门笔记标签统计")
        if top_tags is not None and not top_tags.empty and {'tag', 'tag_count'}.issubset(top_tags.columns):
            max_show = min(50, len(top_tags))
            num_tags_to_show = st.slider(
                "选择显示标签数量:",
                5,
                max_show,
                10,
                key="tags_slider_db"
            )
            fig_tags = px.bar(
                top_tags.head(num_tags_to_show).sort_values('tag_count', ascending=True),
                x='tag_count',
                y='tag',
                orientation='h',
                title=f"Top {num_tags_to_show} 笔记标签",
                labels={'tag_count': '出现次数', 'tag': '标签'},
                color='tag_count',
                color_continuous_scale=px.colors.sequential.Viridis
            )
            st.plotly_chart(fig_tags, use_container_width=True)
        else:
            st.info("热门标签数据为空、加载失败或列名不正确。")

        # --- 评论按天分布 ---
        st.subheader("🗓️ 评论按天分布")
        comment_daily_dist = db_data.get("comment_daily_dist")
        if comment_daily_dist is not None and not comment_daily_dist.empty and 'comment_date' in comment_daily_dist.columns:
            comment_daily_dist['comment_date'] = pd.to_datetime(comment_daily_dist['comment_date'])
            fig_daily = px.bar(
                comment_daily_dist.sort_values('comment_date'),
                x='comment_date',
                y='num_comments',
                title='每日评论数',
                labels={'comment_date': '日期', 'num_comments': '评论数量'},
                color='num_comments',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_daily, use_container_width=True)
        else:
            st.info("评论按天分布数据为空、加载失败或缺少 'comment_date' 列。")

        # --- 评论地理位置情感分布 ---
        st.subheader("🌍 评论地理位置情感分布 (抽样)")
        sentiment_by_location = db_data.get("sentiment_by_location")
        if sentiment_by_location is not None and not sentiment_by_location.empty and \
           {'ip_location', 'sentiment_category', 'count'}.issubset(sentiment_by_location.columns):

            geo_sentiment_filter = st.selectbox(
                "选择情感类别查看地理分布:",
                ["全部"] + list(sentiment_by_location['sentiment_category'].unique()),
                key="geo_sentiment_filter_db"
            )

            geo_data = sentiment_by_location.copy()
            if geo_sentiment_filter != "全部":
                geo_data = geo_data[geo_data['sentiment_category'] == geo_sentiment_filter]

            if not geo_data.empty:
                location_summary = geo_data.groupby('ip_location')['count'].sum().reset_index()
                location_summary = location_summary.sort_values('count', ascending=False)

                max_locations = min(20, len(location_summary))
                top_n_locations = st.slider(
                    "显示评论最多的地区数量:",
                    3,
                    max_locations,
                    10,
                    key="geo_loc_slider_db"
                )

                top_locations = location_summary.head(top_n_locations)['ip_location'].tolist()
                filtered_geo_plot = geo_data[geo_data['ip_location'].isin(top_locations)]

                if not filtered_geo_plot.empty:
                    title_geo = f"Top {top_n_locations} 地区评论情感分布"
                    if geo_sentiment_filter != "全部":
                        title_geo += f" ({geo_sentiment_filter})"

                    fig_geo = px.bar(
                        filtered_geo_plot,
                        x='ip_location',
                        y='count',
                        color='sentiment_category' if geo_sentiment_filter == "全部" else None,
                        color_discrete_map=SENTIMENT_COLORS if geo_sentiment_filter == "全部" else None,
                        barmode='stack',
                        title=title_geo,
                        labels={'ip_location': '地区 (IP属地)', 'count': '评论数量', 'sentiment_category': '情感类别'}
                    )
                    fig_geo.update_xaxes(categoryorder='total descending')
                    st.plotly_chart(fig_geo, use_container_width=True)
                else:
                    st.info(f'在 Top{top_n_locations} 地区中，无 "{geo_sentiment_filter}" 类别数据。')
            else:
                st.info(f'没有 "{geo_sentiment_filter}" 类别的评论数据。')

            st.markdown(
                "<small>*注意：地理分布仅根据IP属地字符串统计。如需地图可视化，应将省份转换为经纬度，"
                "可借助地理编码服务。*</small>",
                unsafe_allow_html=True
            )
        else:
            st.info("地理情感分布数据为空或不完整，无法渲染图表。")

        # --- 笔记点赞/评论统计表 ---
        st.subheader("📝 笔记点赞 & 评论统计 (笔记表 vs 评论表)")
        note_stats = db_data.get("note_stats")
        if note_stats is not None and not note_stats.empty:
            st.dataframe(
                note_stats[['note_id', 'title', 'liked_count', 'note_declared_comment_count', 'actual_comment_count']].head(20),
                height=300,
                use_container_width=True
            )
        else:
            st.info("笔记统计数据为空或加载失败。")
    else:
        st.info("数据库分析数据加载失败。请确保已运行 `python src/main.py db_sql` 并生成结果文件。")

# --- 使用说明 ---
with st.expander("📖 使用说明"):
    st.markdown(f"""
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
    """)

st.markdown("---")
st.markdown("💡 **社交媒体舆情分析系统** | 构建工具：PySpark + SnowNLP + MySQL + Streamlit")
