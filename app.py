import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from scipy import stats
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 设置页面配置
st.set_page_config(
    page_title="高级数据分析助手",
    page_icon="📊",
    layout="wide"
)

# 设置页面标题
st.title("📊 高级数据分析助手")
st.markdown("---")

# 文件上传
uploaded_file = st.file_uploader("选择CSV文件", type=['csv'])

def perform_statistical_tests(df, column):
    """执行统计检验"""
    # 正态性检验
    stat, p_value = stats.normaltest(df[column].dropna())
    return {
        "正态性检验 p值": p_value,
        "偏度": stats.skew(df[column].dropna()),
        "峰度": stats.kurtosis(df[column].dropna())
    }

def create_distribution_plot(df, column):
    """创建分布图"""
    fig = ff.create_distplot(
        [df[column].dropna()],
        [column],
        show_hist=True,
        show_rug=False
    )
    return fig

def perform_anova(df, numeric_col, category_col):
    """执行单因素方差分析"""
    groups = [group for name, group in df.groupby(category_col)[numeric_col]]
    f_stat, p_value = stats.f_oneway(*groups)
    return f_stat, p_value

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"读取文件时出错：{str(e)}")
        st.stop()
    
    with st.sidebar:
        st.header("数据集信息")
        st.write(f"行数: {df.shape[0]}")
        st.write(f"列数: {df.shape[1]}")
        
        # 分类数据和数值数据
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        analysis_type = st.selectbox(
            "选择分析类型",
            ["基础分析", "统计检验", "回归分析", "主成分分析", "时间序列分析"]
        )
        
    # 主要内容区域
    if analysis_type == "基础分析":
        tabs = st.tabs(["数据预览", "描述性统计", "数据可视化"])
        
        with tabs[0]:
            st.subheader("数据预览")
            st.dataframe(df.head(10))
            
            # 数据清洗选项
            if st.checkbox("显示缺失值信息"):
                missing_data = df.isnull().sum()
                st.write("缺失值统计：", missing_data[missing_data > 0])
            
            if st.checkbox("删除缺失值"):
                df = df.dropna()
                st.write("删除缺失值后的数据形状：", df.shape)
        
        with tabs[1]:
            st.subheader("描述性统计")
            st.dataframe(df.describe(include='all'))
            
            # 添加分组统计
            if categorical_columns:
                group_by = st.selectbox("选择分组变量", categorical_columns)
                if group_by:
                    st.write("分组统计：")
                    st.dataframe(df.groupby(group_by).describe())
        
        with tabs[2]:
            st.subheader("数据可视化")
            plot_type = st.selectbox(
                "选择图表类型",
                ["散点图", "箱线图", "直方图", "相关性热力图", "小提琴图"]
            )
            
            if plot_type == "散点图":
                x_col = st.selectbox("选择X轴变量", numeric_columns)
                y_col = st.selectbox("选择Y轴变量", numeric_columns)
                color_col = st.selectbox("选择颜色变量(可选)", ["None"] + categorical_columns)
                
                fig = px.scatter(
                    df,
                    x=x_col,
                    y=y_col,
                    color=None if color_col == "None" else color_col,
                    title=f"{x_col} vs {y_col}"
                )
                st.plotly_chart(fig)
                
            elif plot_type == "箱线图":
                numeric_col = st.selectbox("选择数值变量", numeric_columns)
                category_col = st.selectbox("选择分组变量", categorical_columns)
                fig = px.box(df, x=category_col, y=numeric_col)
                st.plotly_chart(fig)
                
            elif plot_type == "相关性热力图":
                corr = df[numeric_columns].corr()
                fig = px.imshow(
                    corr,
                    title="相关性热力图",
                    color_continuous_scale="RdBu"
                )
                st.plotly_chart(fig)
    
    elif analysis_type == "统计检验":
        st.subheader("统计检验")
        test_type = st.selectbox(
            "选择检验类型",
            ["正态性检验", "t检验", "方差分析", "相关性分析"]
        )
        
        if test_type == "正态性检验":
            col = st.selectbox("选择要检验的变量", numeric_columns)
            results = perform_statistical_tests(df, col)
            st.write(results)
            fig = create_distribution_plot(df, col)
            st.plotly_chart(fig)
            
        elif test_type == "方差分析":
            numeric_col = st.selectbox("选择数值变量", numeric_columns)
            category_col = st.selectbox("选择分组变量", categorical_columns)
            f_stat, p_value = perform_anova(df, numeric_col, category_col)
            st.write(f"F统计量: {f_stat:.4f}")
            st.write(f"P值: {p_value:.4f}")
            
    elif analysis_type == "回归分析":
        st.subheader("回归分析")
        dependent_var = st.selectbox("选择因变量", numeric_columns)
        independent_vars = st.multiselect("选择自变量", numeric_columns)
        
        if independent_vars:
            X = sm.add_constant(df[independent_vars])
            y = df[dependent_var]
            model = sm.OLS(y, X).fit()
            st.write(model.summary())
            
    elif analysis_type == "主成分分析":
        st.subheader("主成分分析")
        pca_vars = st.multiselect("选择变量", numeric_columns)
        
        if pca_vars:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df[pca_vars])
            pca = PCA()
            pca_result = pca.fit_transform(scaled_data)
            
            # 展示解释方差比
            explained_variance_ratio = pca.explained_variance_ratio_
            cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
            
            fig = px.line(
                x=range(1, len(explained_variance_ratio) + 1),
                y=cumulative_variance_ratio,
                title="主成分累积解释方差比",
                labels={"x": "主成分数量", "y": "累积解释方差比"}
            )
            st.plotly_chart(fig)
            
    elif analysis_type == "时间序列分析":
        st.subheader("时间序列分析")
        st.info("请确保数据包含日期列")
        
        date_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
        if not date_columns:
            st.warning("未检测到日期列，请确保日期列格式正确")
        else:
            date_col = st.selectbox("选择日期列", date_columns)
            value_col = st.selectbox("选择值列", numeric_columns)
            
            # 时间序列图
            fig = px.line(
                df,
                x=date_col,
                y=value_col,
                title=f"{value_col}随时间变化趋势"
            )
            st.plotly_chart(fig)

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
