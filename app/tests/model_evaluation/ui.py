import streamlit as st
import pandas as pd
from model_evaluation import get_video_paginator
import plotly.express as px
from typing import Dict, List
import json
import os
from datetime import datetime
from visualization_app import show_visualization

def init_session_state():
    """初始化session state"""
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = {
            'total_videos': 0,
            'processed_videos': 0,
            'total_tags': 0,
            'correct_tags': 0,
            'wrong_tags': 0,  # 错打标签数
            'missed_tags': 0,  # 少打标签数
            'accuracy_by_type': {},
            'tag_distribution': {},
            'error_cases': []
        }
    
    if 'current_video_index' not in st.session_state:
        st.session_state.current_video_index = 0
        
    if 'video_cache' not in st.session_state:
        st.session_state.video_cache = []
        
    if 'paginator' not in st.session_state:
        st.session_state.paginator = get_video_paginator(page_size=100)

def load_current_video():
    """加载当前视频"""
    try:
        # 如果当前索引超出缓存范围，尝试加载下一页
        while st.session_state.current_video_index >= len(st.session_state.video_cache):
            try:
                next_page = next(st.session_state.paginator)
                if not next_page:  # 如果没有更多数据
                    return None
                st.session_state.video_cache.extend(next_page)
            except StopIteration:  # 如果没有更多页
                return None
            
        if 0 <= st.session_state.current_video_index < len(st.session_state.video_cache):
            return st.session_state.video_cache[st.session_state.current_video_index]
        return None
    except Exception as e:
        st.error(f"加载视频失败: {str(e)}")
        return None

def display_video_player(video: Dict):
    """显示视频播放器"""
    if not video:
        st.warning("没有可用的视频")
        return
    
    # 视频播放区
    video_path = video.get('path', '')
    if video_path:
        if video_path.startswith(('http://', 'https://')):
            proxy_url = f"http://10.66.8.56:30501/vision-analyze/video/proxy/{video_path}"
            st.video(proxy_url)
        else:
            try:
                with open(video_path, 'rb') as video_file:
                    video_bytes = video_file.read()
                    st.video(video_bytes)
            except Exception as e:
                st.error(f"无法加载视频文件: {str(e)}")
    else:
        st.warning("视频路径不可用")
    
    # 视频信息区域（使用expander使其可折叠）
    with st.expander("视频详细信息", expanded=False):
        st.write(f"**ID:** {video.get('m_id', 'N/A')}")
        st.write(f"**标题:** {video.get('title', 'Untitled')}")
        
        # 标签信息
        st.write("**当前标签:**")
        tags = video.get('tags', [])
        if tags:
            for tag in tags:
                st.markdown(f"- {tag}")
        else:
            st.write("*无标签*")
        
        # 缩略图
        if video.get('thumbnail_path'):
            st.write("**缩略图:**")
            st.image(video['thumbnail_path'], use_container_width=True)
        
        # 摘要信息
        if video.get('summary_txt'):
            st.write("**视频摘要:**")
            st.markdown(video['summary_txt'])

def get_all_available_tags():
    """获取所有可用的标签列表"""
    return {
        "一、车辆危险行为": {
            "description": "需要精确到秒",
            "tags": {
                "D1: 急刹车": ["车辆突然减速", "刹车灯突然亮起", "车身明显前倾"],
                "D2: 急加速": ["车辆突然提速", "与前车距离快速拉开"],
                "D3: 危险变道": ["变道间距过小", "强行加塞", "未打转向灯变道"],
                "D4: 闯红灯": ["红灯亮起时通过路口", "未在停止线前停车"]
            }
        },
        "二、违规行为": {
            "description": "需要记录持续时间",
            "tags": {
                "V1: 压线行驶": ["轮胎持续压在车道线上", "持续时间超过3秒"],
                "V2: 逆向行驶": ["在单向道逆行", "占用对向车道"],
                "V3: 违规停车": ["在禁停区域停车", "占用应急车道停车"],
                "V4: 未开车灯": ["夜间或雨天未开启车灯", "能见度低时未开启雾灯"]
            }
        },
        "三、交通参与者": {
            "description": "需要记录出现位置和时间",
            "tags": {
                "P1: 行人横穿": ["非斑马线横穿", "闯红灯通过路口"],
                "P2: 非机动车违规": ["电动车逆行", "自行车闯红灯"],
                "P3: 特殊车辆": ["救护车", "消防车", "工程车", "校车"]
            }
        },
        "四、道路环境": {
            "description": "需要记录整个视频区间",
            "tags": {
                "道路类型": ["E1: 高速公路", "E2: 城市主干道", "E3: 乡村道路"],
                "天气状况": ["E4: 晴天", "E5: 阴天", "E6: 雨雪天气", "E7: 雾霾天气"],
                "时间段": ["E8: 白天", "E9: 夜间", "E10: 黎明/黄昏"],
                "特殊情况": ["E11: 拥堵", "E12: 施工", "E13: 事故现场"]
            }
        }
    }

def save_evaluation_record(video: Dict, evaluation_results: Dict):
    """保存评测记录到jsonl文件"""
    # 确保评测数据目录存在
    data_dir = "evaluation_data"
    os.makedirs(data_dir, exist_ok=True)
    
    # 评测记录文件路径
    record_file = os.path.join(data_dir, "evaluation_records.jsonl")
    
    # 构建评测记录
    record = {
        'timestamp': datetime.now().isoformat(),
        'video_info': {
            'id': video.get('m_id'),
            'title': video.get('title'),
            'path': video.get('path'),
            'thumbnail_path': video.get('thumbnail_path'),
            'summary_txt': video.get('summary_txt')
        },
        'original_tags': video.get('tags', []),
        'evaluation_results': evaluation_results
    }
    
    try:
        # 以追加模式写入jsonl文件
        with open(record_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    except Exception as e:
        st.error(f"保存评测记录失败: {str(e)}")

def clean_tag_format(tag):
    """清理标签格式，去除前缀"""
    if ': ' in tag:
        return tag.split(': ')[1]
    return tag

def evaluate_current_video(video: Dict):
    """评测当前视频"""
    st.subheader("评测区域")
    st.markdown("---")
    
    # 安全地获取标签列表
    if not video:
        current_tags = []
    else:
        tags = video.get('tags') or []
        current_tags = [clean_tag_format(tag) for tag in tags]
    
    video_id = video.get('m_id', '') if video else ''  # 安全地获取视频ID
    
    # 创建标签计数器，用于处理重复标签
    tag_counter = {}
    
    # 创建两列布局
    col1, col2 = st.columns(2)
    
    with col1:
        # 错打标签评估区域
        st.markdown("### 错打标签评估")
        
        # 使用container和样式创建视觉边界
        with st.container():
            wrong_tags = []  # 记录错打的标签
            
            # 如果没有标签，显示提示信息
            if not current_tags:
                st.info("当前视频没有标签需要评估")
            else:
                # 评估现有标签
                for i, tag in enumerate(current_tags):
                    # 更新标签计数
                    if tag not in tag_counter:
                        tag_counter[tag] = 0
                    else:
                        tag_counter[tag] += 1
                    
                    # 生成唯一的标签key，包含视频ID
                    tag_key = f"{video_id}_{tag}_{tag_counter[tag]}"
                    
                    # 标签评估区域
                    is_wrong = not st.checkbox("标签正确", key=f"wrong_tag_{tag_key}")
                    st.markdown(f"**{tag}** (#{tag_counter[tag] + 1})")
                    
                    if is_wrong:
                        wrong_tags.append({
                            'tag': tag,
                            'index': tag_counter[tag]
                        })
                        # 如果是需要时间点的标签，添加时间点输入
                        if tag.startswith("D") or tag.startswith("P"):
                            st.number_input("发生时间点(秒)", 0, key=f"time_{tag_key}")
                        # 如果是需要持续时间的标签
                        elif tag.startswith("V"):
                            st.number_input("开始时间(秒)", 0, key=f"start_{tag_key}")
                            st.number_input("结束时间(秒)", 0, key=f"end_{tag_key}")
                    
                    # 如果不是最后一个标签，添加分隔线
                    if i < len(current_tags) - 1:
                        st.markdown("---")
            
            # 自动保存到session state
            st.session_state.wrong_tags = wrong_tags

    with col2:
        # 少打标签补充区域
        with st.container():
            st.markdown("### 少打标签补充")
            
            # 获取所有可用标签
            all_tags = get_all_available_tags()
            missed_tags = []  # 记录少打的标签
            
            # 使用expander为每个大类创建可折叠区域
            for category, category_info in all_tags.items():
                with st.expander(f"{category} ({category_info['description']})"):
                    # 显示每个子类别
                    for main_tag, sub_items in category_info['tags'].items():
                        # 对于道路环境类别，特殊处理
                        if category == "四、道路环境":
                            # 显示分组标题
                            st.markdown(f"**{main_tag}**")
                            # 显示具体标签项
                            for sub_tag in sub_items:
                                clean_sub_tag = clean_tag_format(sub_tag)
                                if clean_sub_tag not in current_tags:
                                    st.markdown('<div style="margin: 10px 0; padding-left: 20px;">', unsafe_allow_html=True)
                                    
                                    # 显示具体标签和选择框
                                    if st.checkbox(clean_sub_tag, key=f"missed_tag_{video_id}_{clean_sub_tag}"):
                                        missed_tags.append(clean_sub_tag)
                                    
                                    st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            # 其他类别保持原有逻辑
                            clean_main_tag = clean_tag_format(main_tag)
                            if clean_main_tag not in current_tags:
                                st.markdown('<div style="margin: 10px 0;">', unsafe_allow_html=True)
                                
                                # 显示主标签和其描述
                                if st.checkbox(clean_main_tag, key=f"missed_tag_{video_id}_{clean_main_tag}"):
                                    missed_tags.append(clean_main_tag)
                                    
                                    # 只有选中标签时才显示时间输入
                                    if clean_main_tag.startswith(("D", "P")):
                                        st.number_input("时间点(秒)", 0, key=f"time_{video_id}_{clean_main_tag}")
                                    elif clean_main_tag.startswith("V"):
                                        st.number_input("持续时间(秒)", 0, key=f"duration_{video_id}_{clean_main_tag}")
                                
                                # 显示描述作为帮助信息
                                if sub_items:
                                    st.markdown("*标签说明:*")
                                    for desc in sub_items:
                                        st.markdown(f"- {desc}")
                                
                                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # 评测结果摘要区域
    st.markdown("---")
    st.markdown("### 评测结果摘要")
    with st.form("evaluation_form"):
        st.write(f"- 现有标签数: {len(current_tags)}")
        st.write(f"- 错打标签数: {len(st.session_state.get('wrong_tags', []))}")
        st.write(f"- 少打标签数: {len(missed_tags)}")
        
        # 添加评测备注
        notes = st.text_area("评测备注")
        
        # 提交按钮
        submitted = st.form_submit_button("提交评测")
        
        if submitted:
            # 收集评测结果
            results = {
                'video_id': video['m_id'],
                'total_tags': len(current_tags),
                'correct_tags': len(current_tags) - len(st.session_state.wrong_tags),
                'wrong_tags': [{
                    'tag': wrong_tag['tag'],
                    'index': wrong_tag['index'],
                    'time': st.session_state.get(f"time_{wrong_tag['tag']}_{wrong_tag['index']}"),
                    'start': st.session_state.get(f"start_{wrong_tag['tag']}_{wrong_tag['index']}"),
                    'end': st.session_state.get(f"end_{wrong_tag['tag']}_{wrong_tag['index']}")
                } for wrong_tag in st.session_state.wrong_tags],
                'missed_tags': [{
                    'tag': tag,
                    'time': st.session_state.get(f"time_{tag}"),
                    'duration': st.session_state.get(f"duration_{tag}")
                } for tag in missed_tags],
                'notes': notes,
                'evaluation_time': datetime.now().isoformat()
            }
            
            # 更新评测统计
            st.session_state.evaluation_results['processed_videos'] += 1
            st.session_state.evaluation_results['total_tags'] += results['total_tags']
            st.session_state.evaluation_results['correct_tags'] += results['correct_tags']
            st.session_state.evaluation_results['wrong_tags'] += len(st.session_state.wrong_tags)
            st.session_state.evaluation_results['missed_tags'] += len(missed_tags)
            
            # 保存评测记录
            save_evaluation_record(video, results)
            
            st.success("评测完成!")
            return results
    
    return None

def clear_evaluation_states():
    """清除评测相关的状态"""
    # 获取所有session state的key
    keys_to_remove = []
    for key in st.session_state:
        # 清除标签相关的状态
        if key.startswith(('wrong_tag_', 'time_', 'start_', 'end_', 'missed_tag_', 'duration_')):
            keys_to_remove.append(key)
    
    # 删除这些key
    for key in keys_to_remove:
        del st.session_state[key]
    
    # 清除wrong_tags状态
    st.session_state.wrong_tags = []
    # 清除missed_tags状态
    st.session_state.missed_tags = []

def display_navigation():
    """显示导航控制"""
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col1:
        if st.button("上一个", disabled=st.session_state.current_video_index == 0):
            # 清除评测状态
            clear_evaluation_states()
            # 更新视频索引
            st.session_state.current_video_index -= 1
            st.rerun()
            
    with col2:
        st.write(f"当前第 {st.session_state.current_video_index + 1} 个视频")
        
    with col3:
        # 尝试预加载下一页，检查是否还有更多视频
        has_more = False
        if st.session_state.current_video_index >= len(st.session_state.video_cache) - 1:
            try:
                # 临时获取下一页数据来检查是否还有更多
                next_page = next(st.session_state.paginator)
                if next_page:
                    has_more = True
            except StopIteration:
                has_more = False
        else:
            has_more = True
            
        if st.button("下一个", disabled=not has_more):
            # 清除评测状态
            clear_evaluation_states()
            # 更新视频索引
            st.session_state.current_video_index += 1
            st.rerun()

def export_results():
    """导出评测结果"""
    results = st.session_state.evaluation_results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 生成评估报告数据
    evaluation_report = {
        "评估时间": timestamp,
        "总体统计": {
            "总视频数": 100,
            "已评估视频数": results['processed_videos'],
            "总标签数": results['total_tags'],
            "正确标签数": results['correct_tags'],
            "错误标签数": results['wrong_tags'],
            "遗漏标签数": results['missed_tags']
        },
        "准确率分析": {
            "标签准确率": f"{(results['correct_tags'] / results['total_tags'] * 100):.2f}%" if results['total_tags'] > 0 else "0%",
            "错打率": f"{(results['wrong_tags'] / results['total_tags'] * 100):.2f}%" if results['total_tags'] > 0 else "0%",
            "平均每视频错误标签数": f"{(results['wrong_tags'] / results['processed_videos']):.2f}" if results['processed_videos'] > 0 else "0",
            "平均每视频遗漏标签数": f"{(results['missed_tags'] / results['processed_videos']):.2f}" if results['processed_videos'] > 0 else "0"
        },
        "标签分布": results.get('tag_distribution', {}),
        "错误分析": {
            "常见错误类型": results.get('error_cases', []),
            "标签类型准确率": results.get('accuracy_by_type', {})
        },
        "评估建议": {
            "模型改进方向": [
                "提高对复杂场景的识别能力",
                "优化对特定标签类型的识别",
                "减少误报和漏报"
            ],
            "评估改进建议": [
                "增加样本多样性",
                "扩大评估数据集规模",
                "细化错误类型分析"
            ]
        }
    }
    
    # 保存评估报告
    report_filename = f"evaluation_report_{timestamp}.json"
    with open(report_filename, 'w', encoding='utf-8') as f:
        json.dump(evaluation_report, f, ensure_ascii=False, indent=2)
    
    # 保存原始结果
    results_filename = f"evaluation_results_{timestamp}.json"
    with open(results_filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 提供下载按钮
    with open(report_filename, 'r', encoding='utf-8') as f:
        st.download_button(
            label="下载评估报告",
            data=f.read(),
            file_name=report_filename,
            mime="application/json"
        )

def display_evaluation_stats():
    """显示评测统计信息"""
    st.sidebar.header("评测统计")
    
    # 基本统计
    st.sidebar.metric("已评测视频数", st.session_state.evaluation_results['processed_videos'])
    st.sidebar.metric("总标签数", st.session_state.evaluation_results['total_tags'])
    
    # 准确率统计
    if st.session_state.evaluation_results['total_tags'] > 0:
        # 标签准确率
        accuracy = (st.session_state.evaluation_results['correct_tags'] / 
                   st.session_state.evaluation_results['total_tags'] * 100)
        st.sidebar.metric("标签准确率", f"{accuracy:.2f}%")
        
        # 错打率
        wrong_rate = (st.session_state.evaluation_results['wrong_tags'] / 
                     st.session_state.evaluation_results['total_tags'] * 100)
        st.sidebar.metric("错打率", f"{wrong_rate:.2f}%")
        
        # 少打数
        st.sidebar.metric("累计少打标签数", st.session_state.evaluation_results['missed_tags'])

def main():
    # 设置页面配置必须是第一个命令
    st.set_page_config(page_title="Qwen模型评测工具", layout="wide")
    
    # 添加页面选择
    page = st.sidebar.radio(
        "选择功能",
        ["评测界面", "评测报告"],
        index=0
    )
    
    if page == "评测界面":
        show_evaluation_ui()
    else:
        show_visualization()

def show_evaluation_ui():
    st.title("Qwen视觉模型评测工具")
    
    # 初始化session state
    init_session_state()
    
    # 加载当前视频
    current_video = load_current_video()
    
    if current_video:
        # 显示导航
        display_navigation()
        
        # 创建主要布局
        video_col, evaluation_col = st.columns([1, 1])
        
        with video_col:
            # 视频播放和信息区域
            display_video_player(current_video)
        
        with evaluation_col:
            # 评测区域
            evaluate_current_video(current_video)
        
        # 侧边栏统计信息
        display_evaluation_stats()
        
        # 导出报告按钮
        if st.session_state.evaluation_results['processed_videos'] > 0:
            st.sidebar.button("导出报告", on_click=export_results)
    else:
        st.error("无法加载视频数据")

def show_visualization():
    st.title("评估报告可视化")
    
    # 显示可视化图表
    from visualization import generate_evaluation_report
    
    # 确保目录存在
    output_path = "./model_evaluation/reports"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # 生成报告
    jsonl_path = "./evaluation_data/evaluation_records.jsonl"
    report = generate_evaluation_report(jsonl_path, output_path)
    
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
    total_tags = report['total_statistics']['total_tags']
    if total_tags > 0:
        accuracy = (report['total_statistics']['correct_tags'] / total_tags * 100)
        st.metric("总体准确率", f"{accuracy:.1f}%")
    else:
        st.metric("总体准确率", "暂无数据")
    
    # 显示图表
    st.subheader("评测结果可视化")
    
    # 创建三个标签页
    tab1, tab2, tab3 = st.tabs(["总体分布", "准确率分析", "召回率分析"])
    
    # 总体分布标签页
    with tab1:
        try:
            # 显示图表
            with open(f"{output_path}/overall_accuracy.html", "r", encoding='utf-8') as f:
                data = f.read()
                st.components.v1.html(data, height=800)
            
            # 创建下载按钮的列布局
            col1, col2 = st.columns(2)
            
            # 图表下载按钮
            with col1:
                st.download_button(
                    label="下载总体分布图表",
                    data=data,
                    file_name="overall_accuracy.html",
                    mime="text/html"
                )
            
            # JSON数据下载按钮
            with col2:
                overall_data = {
                    "total_statistics": report['total_statistics'],
                    "distribution": {
                        "correct_tags": report['total_statistics']['correct_tags'],
                        "wrong_tags": report['total_statistics']['wrong_tags'],
                        "missed_tags": report['total_statistics']['missed_tags']
                    }
                }
                st.download_button(
                    label="下载总体分布数据",
                    data=json.dumps(overall_data, ensure_ascii=False, indent=2),
                    file_name="overall_distribution.json",
                    mime="application/json"
                )
        except FileNotFoundError:
            st.warning("总体准确率图表文件不存在")
    
    # 准确率分析标签页
    with tab2:
        try:
            # 显示图表
            with open(f"{output_path}/tag_accuracy.html", "r", encoding='utf-8') as f:
                data = f.read()
                st.components.v1.html(data, height=1000)
            
            # 创建下载按钮的列布局
            col1, col2 = st.columns(2)
            
            # 图表下载按钮
            with col1:
                st.download_button(
                    label="下载准确率分析图表",
                    data=data,
                    file_name="tag_accuracy.html",
                    mime="text/html"
                )
            
            # JSON数据下载按钮
            with col2:
                accuracy_data = {
                    "overall_accuracy": f"{(report['total_statistics']['correct_tags']/report['total_statistics']['total_tags']*100):.1f}%" if report['total_statistics']['total_tags'] > 0 else "暂无数据",
                    "tag_accuracy": report.get('tag_statistics', {}),
                    "accuracy_by_type": report.get('accuracy_by_type', {})
                }
                st.download_button(
                    label="下载准确率分析数据",
                    data=json.dumps(accuracy_data, ensure_ascii=False, indent=2),
                    file_name="accuracy_analysis.json",
                    mime="application/json"
                )
        except FileNotFoundError:
            st.warning("标签准确率图表文件不存在")
            
    # 召回率分析标签页
    with tab3:
        try:
            # 显示图表
            with open(f"{output_path}/tag_recall.html", "r", encoding='utf-8') as f:
                data = f.read()
                st.components.v1.html(data, height=1000)
            
            # 创建下载按钮的列布局
            col1, col2 = st.columns(2)
            
            # 图表下载按钮
            with col1:
                st.download_button(
                    label="下载召回率分析图表",
                    data=data,
                    file_name="tag_recall.html",
                    mime="text/html"
                )
            
            # JSON数据下载按钮
            with col2:
                total_videos = report['total_statistics']['total_videos']
                recall_data = {
                    "tag_recall": report.get('tag_statistics', {}),
                    "missed_tags_analysis": {
                        "total_missed": report['total_statistics']['missed_tags'],
                        "missed_by_type": report.get('missed_tags_by_type', {}),
                        "average_missed_per_video": report['total_statistics']['missed_tags'] / total_videos if total_videos > 0 else 0
                    }
                }
                st.download_button(
                    label="下载召回率分析数据",
                    data=json.dumps(recall_data, ensure_ascii=False, indent=2),
                    file_name="recall_analysis.json",
                    mime="application/json"
                )
        except FileNotFoundError:
            st.warning("标签召回率图表文件不存在")

if __name__ == "__main__":
    main() 