import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns

# 设置页面配置
st.set_page_config(
    page_title="数据分析助手",
    page_icon="📊",
    layout="wide"
)

# 设置页面标题
st.title("📊 数据分析助手")
st.markdown("---")

# 文件上传
uploaded_file = st.file_uploader("选择CSV文件", type=['csv'])

if uploaded_file is not None:
    try:
        # 读取数据
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"读取文件时出错：{str(e)}")
        st.stop()
    
    # 侧边栏 - 数据基本信息
    with st.sidebar:
        st.header("数据集信息")
        st.write(f"行数: {df.shape[0]}")
        st.write(f"列数: {df.shape[1]}")
        st.write("列名:", df.columns.tolist())
        
        # 选择要分析的列
        selected_columns = st.multiselect(
            "选择要分析的列",
            df.columns.tolist(),
            default=df.select_dtypes(include=[np.number]).columns.tolist()[:2]
        )
    
    # 主要内容区域
    tabs = st.tabs(["数据预览", "描述性统计", "数据可视化"])
    
    # 数据预览标签页
    with tabs[0]:
        st.subheader("数据预览")
        st.dataframe(df.head(10))
        
    # 描述性统计标签页
    with tabs[1]:
        st.subheader("描述性统计")
        st.dataframe(df.describe())
        
    # 数据可视化标签页
    with tabs[2]:
        st.subheader("数据可视化")
        if len(selected_columns) >= 2:
            # 散点图
            fig = px.scatter(
                df,
                x=selected_columns[0],
                y=selected_columns[1],
                title=f"{selected_columns[0]} vs {selected_columns[1]}"
            )
            st.plotly_chart(fig)
            
            # 相关性热力图
            if len(selected_columns) > 2:
                st.subheader("相关性热力图")
                corr = df[selected_columns].corr()
                fig = px.imshow(
                    corr,
                    title="相关性热力图",
                    color_continuous_scale="RdBu"
                )
                st.plotly_chart(fig)
        else:
            st.warning("请至少选择两列进行可视化分析")

else:
    st.info("请上传CSV文件开始分析")
    
    # 示例数据
    if st.button("使用示例数据"):
        # 创建示例数据
        np.random.seed(42)
        data = {
            '销售额': np.random.normal(1000, 200, 100),
            '利润': np.random.normal(200, 50, 100),
            '客户数': np.random.randint(10, 100, 100),
            '地区': np.random.choice(['北部', '南部', '东部', '西部'], 100)
        }
        df = pd.DataFrame(data)
        
        # 直接使用示例数据，无需保存为临时文件
        uploaded_file = True  # 设置为 True 来触发数据分析逻辑
        
        # 移动侧边栏代码到这里
        with st.sidebar:
            st.header("数据集信息")
            st.write(f"行数: {df.shape[0]}")
            st.write(f"列数: {df.shape[1]}")
            st.write("列名:", df.columns.tolist())
            
            # 选择要分析的列
            selected_columns = st.multiselect(
                "选择要分析的列",
                df.columns.tolist(),
                default=df.select_dtypes(include=[np.number]).columns.tolist()[:2]
            )
        
        # 主要内容区域
        tabs = st.tabs(["数据预览", "描述性统计", "数据可视化"])
        
        # 数据预览标签页
        with tabs[0]:
            st.subheader("数据预览")
            st.dataframe(df.head(10))
            
        # 描述性统计标签页
        with tabs[1]:
            st.subheader("描述性统计")
            st.dataframe(df.describe())
            
        # 数据可视化标签页
        with tabs[2]:
            st.subheader("数据可视化")
            if len(selected_columns) >= 2:
                # 散点图
                fig = px.scatter(
                    df,
                    x=selected_columns[0],
                    y=selected_columns[1],
                    title=f"{selected_columns[0]} vs {selected_columns[1]}"
                )
                st.plotly_chart(fig)
                
                # 相关性热力图
                if len(selected_columns) > 2:
                    st.subheader("相关性热力图")
                    corr = df[selected_columns].corr()
                    fig = px.imshow(
                        corr,
                        title="相关性热力图",
                        color_continuous_scale="RdBu"
                    )
                    st.plotly_chart(fig)
            else:
                st.warning("请至少选择两列进行可视化分析")
