import streamlit as st
import streamlit.components.v1 as components
from visualization import generate_evaluation_report
import os

def show_visualization():
    """显示可视化报告"""
    st.title("模型评测可视化报告")
    
    # 生成评测报告
    jsonl_path = "./evaluation_data/evaluation_records.jsonl"
    output_path = "./reports"  # 简化输出路径
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    try:
        # 生成报告
        report = generate_evaluation_report(jsonl_path, output_path)
        
        if report['total_statistics']['total_videos'] == 0:
            st.warning("暂无评测数据。请先进行模型评测，生成评测数据后再查看报告。")
            return
            
        # 显示总体统计
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("总视频数", report['total_statistics']['total_videos'])
        with col2:
            st.metric("总标签数", report['total_statistics']['total_tags'])
        with col3:
            st.metric("正确标签数", report['total_statistics']['correct_tags'])
        with col4:
            st.metric("错误标签数", report['total_statistics']['wrong_tags'])
        with col5:
            st.metric("遗漏标签数", report['total_statistics']['missed_tags'])
        
        # 显示准确率
        if report['total_statistics']['total_tags'] > 0:
            st.metric(
                "总体准确率", 
                f"{(report['total_statistics']['correct_tags']/report['total_statistics']['total_tags']*100):.1f}%"
            )
        
        # 显示图表
        st.subheader("评测结果可视化")
        
        tab1, tab2, tab3 = st.tabs(["总体分布", "准确率分析", "召回率分析"])
        
        with tab1:
            components.html(
                open(os.path.join(output_path, "overall_accuracy.html"), 'r', encoding='utf-8').read(),
                height=600
            )
        
        with tab2:
            components.html(
                open(os.path.join(output_path, "tag_accuracy.html"), 'r', encoding='utf-8').read(),
                height=800
            )
        
        with tab3:
            components.html(
                open(os.path.join(output_path, "tag_recall.html"), 'r', encoding='utf-8').read(),
                height=800
            )
    
    except Exception as e:
        if str(e) == "division by zero":
            st.warning("暂无评测数据。请先进行模型评测，生成评测数据后再查看报告。")
        else:
            st.error(f"生成报告时发生错误: {str(e)}")

if __name__ == "__main__":
    st.set_page_config(
        page_title="模型评测报告",
        page_icon="📊",
        layout="wide"
    )
    show_visualization() 