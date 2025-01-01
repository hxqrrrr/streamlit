import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from scipy import stats
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class DataAnalyzer:
    def __init__(self, df):
        self.df = df
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()
        self.date_columns = df.select_dtypes(include=['datetime64']).columns.tolist()

    def perform_statistical_tests(self, column):
        """执行统计检验"""
        stat, p_value = stats.normaltest(self.df[column].dropna())
        return {
            "正态性检验 p值": p_value,
            "偏度": stats.skew(self.df[column].dropna()),
            "峰度": stats.kurtosis(self.df[column].dropna())
        }

    def create_distribution_plot(self, column):
        """创建分布图"""
        return ff.create_distplot(
            [self.df[column].dropna()],
            [column],
            show_hist=True,
            show_rug=False
        )

    def perform_anova(self, numeric_col, category_col):
        """执行单因素方差分析"""
        groups = [group for name, group in self.df.groupby(category_col)[numeric_col]]
        f_stat, p_value = stats.f_oneway(*groups)
        return f_stat, p_value

    def show_basic_analysis(self):
        """显示基础分析"""
        tabs = st.tabs(["数据预览", "描述性统计", "数据可视化"])
        
        with tabs[0]:
            self._show_data_preview()
        
        with tabs[1]:
            self._show_descriptive_stats()
        
        with tabs[2]:
            self._show_visualizations()

    def _show_data_preview(self):
        """显示数据预览"""
        st.subheader("数据预览")
        st.dataframe(self.df.head(10))
        
        if st.checkbox("显示缺失值信息"):
            missing_data = self.df.isnull().sum()
            st.write("缺失值统计：", missing_data[missing_data > 0])
        
        if st.checkbox("删除缺失值"):
            self.df = self.df.dropna()
            st.write("删除缺失值后的数据形状：", self.df.shape)

    def _show_descriptive_stats(self):
        """显示描述性统计"""
        st.subheader("描述性统计")
        st.dataframe(self.df.describe(include='all'))
        
        if self.categorical_columns:
            group_by = st.selectbox("选择分组变量", self.categorical_columns)
            if group_by:
                st.write("分组统计：")
                st.dataframe(self.df.groupby(group_by).describe())

    def show_statistical_tests(self):
        """显示统计检验"""
        test_type = st.selectbox(
            "选择检验类型",
            ["正态性检验", "t检验", "方差分析", "相关性分析"]
        )
        
        if test_type == "正态性检验":
            col = st.selectbox("选择要检验的变量", self.numeric_columns)
            results = self.perform_statistical_tests(col)
            st.write(results)
            fig = self.create_distribution_plot(col)
            st.plotly_chart(fig)
            
        elif test_type == "方差分析":
            numeric_col = st.selectbox("选择数值变量", self.numeric_columns)
            category_col = st.selectbox("选择分组变量", self.categorical_columns)
            f_stat, p_value = self.perform_anova(numeric_col, category_col)
            st.write(f"F统计量: {f_stat:.4f}")
            st.write(f"P值: {p_value:.4f}")

    def show_regression_analysis(self):
        """显示回归分析"""
        dependent_var = st.selectbox("选择因变量", self.numeric_columns)
        independent_vars = st.multiselect("选择自变量", self.numeric_columns)
        
        if independent_vars:
            X = sm.add_constant(self.df[independent_vars])
            y = self.df[dependent_var]
            model = sm.OLS(y, X).fit()
            st.write(model.summary())

    def show_pca_analysis(self):
        """显示主成分分析"""
        pca_vars = st.multiselect("选择变量", self.numeric_columns)
        
        if pca_vars:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(self.df[pca_vars])
            pca = PCA()
            pca_result = pca.fit_transform(scaled_data)
            
            explained_variance_ratio = pca.explained_variance_ratio_
            cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
            
            fig = px.line(
                x=range(1, len(explained_variance_ratio) + 1),
                y=cumulative_variance_ratio,
                title="主成分累积解释方差比",
                labels={"x": "主成分数量", "y": "累积解释方差比"}
            )
            st.plotly_chart(fig)

    def show_time_series_analysis(self):
        """显示时间序列分析"""
        st.info("请确保数据包含日期列")
        
        if not self.date_columns:
            st.warning("未检测到日期列，请确保日期列格式正确")
        else:
            date_col = st.selectbox("选择日期列", self.date_columns)
            value_col = st.selectbox("选择值列", self.numeric_columns)
            
            fig = px.line(
                self.df,
                x=date_col,
                y=value_col,
                title=f"{value_col}随时间变化趋势"
            )
            st.plotly_chart(fig)

    def _show_visualizations(self):
        """显示数据可视化"""
        st.subheader("数据可视化")
        plot_type = st.selectbox(
            "选择图表类型",
            ["散点图", "箱线图", "直方图", "相关性热力图", "小提琴图", "条形图"]
        )
        
        if plot_type == "散点图":
            x_col = st.selectbox("选择X轴变量", self.numeric_columns)
            y_col = st.selectbox("选择Y轴变量", self.numeric_columns)
            color_col = st.selectbox("选择颜色变量(可选)", ["None"] + self.categorical_columns)
            
            fig = px.scatter(
                self.df,
                x=x_col,
                y=y_col,
                color=None if color_col == "None" else color_col,
                title=f"{x_col} vs {y_col}"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif plot_type == "箱线图":
            numeric_col = st.selectbox("选择数值变量", self.numeric_columns)
            category_col = st.selectbox("选择分组变量", self.categorical_columns)
            fig = px.box(self.df, x=category_col, y=numeric_col)
            st.plotly_chart(fig, use_container_width=True)
            
        elif plot_type == "直方图":
            col = st.selectbox("选择变量", self.numeric_columns)
            bins = st.slider("选择箱数", min_value=5, max_value=100, value=30)
            fig = px.histogram(self.df, x=col, nbins=bins)
            st.plotly_chart(fig, use_container_width=True)
            
        elif plot_type == "相关性热力图":
            if len(self.numeric_columns) > 1:
                corr = self.df[self.numeric_columns].corr()
                fig = px.imshow(
                    corr,
                    title="相关性热力图",
                    color_continuous_scale="RdBu",
                    aspect="auto"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("需要至少两个数值变量来创建相关性热力图")
            
        elif plot_type == "小提琴图":
            numeric_col = st.selectbox("选择数值变量", self.numeric_columns)
            category_col = st.selectbox("选择分组变量", self.categorical_columns)
            fig = px.violin(self.df, x=category_col, y=numeric_col)
            st.plotly_chart(fig, use_container_width=True)
            
        elif plot_type == "条形图":
            if self.categorical_columns:
                category_col = st.selectbox("选择分类变量", self.categorical_columns)
                agg_func = st.selectbox("选择统计方式", ["计数", "求和", "平均值"])
                
                if agg_func == "计数":
                    fig = px.bar(
                        self.df[category_col].value_counts().reset_index(),
                        x="index",
                        y=category_col,
                        title=f"{category_col}的分布"
                    )
                else:
                    numeric_col = st.selectbox("选择数值变量", self.numeric_columns)
                    agg_dict = {"求和": "sum", "平均值": "mean"}
                    agg_data = self.df.groupby(category_col)[numeric_col].agg(agg_dict[agg_func]).reset_index()
                    fig = px.bar(
                        agg_data,
                        x=category_col,
                        y=numeric_col,
                        title=f"{category_col}与{numeric_col}的{agg_func}"
                    )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("需要至少一个分类变量来创建条形图")

    def run(self):
        """运行分析程序"""
        with st.sidebar:
            st.header("数据集信息")
            st.write(f"行数: {self.df.shape[0]}")
            st.write(f"列数: {self.df.shape[1]}")
            
            analysis_type = st.selectbox(
                "选择分析类型",
                ["基础分析", "统计检验", "回归分析", "主成分分析", "时间序列分析"]
            )
        
        if analysis_type == "基础分析":
            self.show_basic_analysis()
        elif analysis_type == "统计检验":
            self.show_statistical_tests()
        elif analysis_type == "回归分析":
            self.show_regression_analysis()
        elif analysis_type == "主成分分析":
            self.show_pca_analysis()
        elif analysis_type == "时间序列分析":
            self.show_time_series_analysis() 