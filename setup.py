from setuptools import setup, find_packages

setup(
    name="social_media_sentiment_analysis",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'pyspark>=3.5.0',
        'pandas',
        'numpy',
        'snownlp',
        'mysql-connector-python',
        'streamlit',
        'plotly',
        'matplotlib',
        'wordcloud'
    ],
    python_requires='>=3.8',
) 