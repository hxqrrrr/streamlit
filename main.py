import streamlit as st
from data_analyzer import DataAnalyzer
import pandas as pd
import numpy as np

def load_data(file):
    """加载不同格式的数据文件"""
    file_type = file.name.split('.')[-1].lower()
    try:
        if file_type == 'csv':
            return pd.read_csv(file)
        elif file_type == 'xlsx' or file_type == 'xls':
            return pd.read_excel(file)
        elif file_type == 'json':
            return pd.read_json(file)
        elif file_type == 'parquet':
            return pd.read_parquet(file)
        else:
            raise ValueError(f"不支持的文件格式: {file_type}")
    except Exception as e:
        st.error(f"读取文件时出错：{str(e)}")
        return None

def create_example_data():
    """创建示例数据"""
    np.random.seed(42)
    data = {
        '销售额': np.random.normal(1000, 200, 100),
        '利润': np.random.normal(200, 50, 100),
        '客户数': np.random.randint(10, 100, 100),
        '地区': np.random.choice(['北部', '南部', '东部', '西部'], 100),
        '日期': pd.date_range(start='2023-01-01', periods=100)
    }
    return pd.DataFrame(data)

def main():
    # 设置页面配置
    st.set_page_config(
        page_title="高级数据分析助手",
        page_icon="📊",
        layout="wide"
    )

    # 设置页面标题
    st.title("📊 高级数据分析助手")
    st.markdown("---")

    # 初始化 session_state
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
        st.session_state.df = None

    # 文件上传
    uploaded_file = st.file_uploader(
        "选择数据文件", 
        type=['csv', 'xlsx', 'xls', 'json', 'parquet']
    )

    if uploaded_file is not None:
        # 只在文件改变时重新加载数据
        file_name = uploaded_file.name
        if 'current_file' not in st.session_state or st.session_state.current_file != file_name:
            df = load_data(uploaded_file)
            if df is not None:
                st.session_state.df = df
                st.session_state.analyzer = DataAnalyzer(df)
                st.session_state.current_file = file_name

    # 使用示例数据按钮
    if st.button("使用示例数据"):
        df = create_example_data()
        st.session_state.df = df
        st.session_state.analyzer = DataAnalyzer(df)
        st.session_state.current_file = "example_data"

    # 运行分析
    if st.session_state.analyzer is not None:
        st.session_state.analyzer.run()
    else:
        st.info("请上传数据文件开始分析")

if __name__ == "__main__":
    main() 