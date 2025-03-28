import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict

def load_evaluation_records(jsonl_path):
    """加载评测记录"""
    records = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))
    return records

def calculate_statistics(records):
    """计算评测统计数据"""
    total_stats = {
        'total_videos': len(records),
        'total_tags': 0,
        'correct_tags': 0,
        'wrong_tags': 0,
        'missed_tags': 0
    }
    
    tag_stats = defaultdict(lambda: {
        'total': 0,        # 模型打的标签总数
        'correct': 0,      # 正确标签数
        'wrong': 0,        # 错打标签数
        'missed': 0,       # 漏打标签数
        'model_tags': 0    # 模型打的标签数(用于计算召回率)
    })
    
    for record in records:
        eval_results = record.get('evaluation_results', {})
        original_tags = record.get('original_tags', []) or []  # 确保是列表
        
        # 累计总数
        total_stats['total_tags'] += eval_results.get('total_tags', 0)
        total_stats['correct_tags'] += eval_results.get('correct_tags', 0)
        total_stats['wrong_tags'] += len(eval_results.get('wrong_tags', []))
        total_stats['missed_tags'] += len(eval_results.get('missed_tags', []))
        
        # 统计模型打的标签数
        for tag in original_tags:
            if tag:  # 确保标签不是None
                tag_stats[tag]['model_tags'] += 1
                tag_stats[tag]['total'] += 1
        
        # 统计错误标签
        wrong_tags = eval_results.get('wrong_tags', []) or []
        for wrong_tag in wrong_tags:
            if isinstance(wrong_tag, dict) and 'tag' in wrong_tag:
                tag = wrong_tag['tag']
            else:
                tag = wrong_tag
            if tag:  # 确保标签不是None
                tag_stats[tag]['wrong'] += 1
            
        # 统计遗漏标签
        missed_tags = eval_results.get('missed_tags', []) or []
        for missed_tag in missed_tags:
            if isinstance(missed_tag, dict) and 'tag' in missed_tag:
                tag = missed_tag['tag']
            else:
                tag = missed_tag
            if tag:  # 确保标签不是None
                tag_stats[tag]['missed'] += 1
                tag_stats[tag]['total'] += 1  # 增加总数，因为这是应该有的标签
        
        # 计算正确标签数
        for tag in tag_stats:
            if tag:  # 确保标签不是None
                tag_stats[tag]['correct'] = tag_stats[tag]['model_tags'] - tag_stats[tag]['wrong']
    
    return total_stats, dict(tag_stats)

def generate_evaluation_report(jsonl_path, output_path):
    """生成评测报告"""
    try:
        records = load_evaluation_records(jsonl_path)
    except Exception as e:
        print(f"加载评测记录失败: {str(e)}")
        records = []
    
    # 检查是否有评测记录
    if not records:
        return {
            'total_statistics': {
                'total_videos': 0,
                'total_tags': 0,
                'correct_tags': 0,
                'wrong_tags': 0,
                'missed_tags': 0
            },
            'tag_statistics': {},
            'tag_analysis': []
        }
    
    try:
        total_stats, tag_stats = calculate_statistics(records)
        
        # 生成总体准确率饼图
        fig_overall = go.Figure(data=[
            go.Pie(
                labels=['正确标签', '错误标签', '遗漏标签'],
                values=[total_stats['correct_tags'], 
                       total_stats['wrong_tags'],
                       total_stats['missed_tags']],
                hole=.3,
                marker=dict(
                    colors=['#2ecc71',  # 绿色表示正确
                           '#e74c3c',   # 红色表示错误
                           '#f1c40f']   # 黄色表示遗漏
                ),
                textinfo='label+percent',  # 显示标签和百分比
                textposition='inside',     # 文字位置
                insidetextorientation='radial'  # 文字方向
            )
        ])
        
        # 更新饼图布局
        fig_overall.update_layout(
            title={
                'text': '标签评测结果分布',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            showlegend=True,
            legend={
                'orientation': 'h',
                'yanchor': 'bottom',
                'y': -0.1,
                'xanchor': 'center',
                'x': 0.5
            },
            width=600,
            height=500,
            margin=dict(t=80, b=80, l=40, r=40)
        )
        
        fig_overall.write_html(f"{output_path}/overall_accuracy.html")
        
        # 准备标签数据
        tag_data = []
        for tag, stats in tag_stats.items():
            if not tag:  # 跳过None或空标签
                continue
                
            # 计算准确率
            accuracy = (stats['correct'] / stats['model_tags'] * 100) if stats['model_tags'] > 0 else 0
            
            # 计算召回率
            recall_denominator = stats['model_tags'] - stats['wrong'] + stats['missed']
            recall_numerator = stats['model_tags'] - stats['wrong']
            recall_rate = (recall_numerator / recall_denominator * 100) if recall_denominator > 0 else 0
            
            tag_data.append({
                'tag': tag,
                'accuracy': accuracy,
                'recall_rate': recall_rate,
                'accuracy_total': stats['model_tags'],  # 准确率的分母
                'recall_total': recall_denominator,     # 召回率的分母
                'correct': stats['correct'],
                'wrong': stats['wrong'],
                'missed': stats['missed'],
                'model_tags': stats['model_tags']
            })
        
        # 按总样本量排序
        tag_data.sort(key=lambda x: x['recall_total'], reverse=True)
        
        # 提取绘图所需的数据
        tags = [d['tag'] for d in tag_data]
        recall_rates = [d['recall_rate'] for d in tag_data]
        sample_sizes = [d['recall_total'] for d in tag_data]
        
        # 生成准确率分析图表
        fig_accuracy = go.Figure()
        
        # 添加准确率条形图
        fig_accuracy.add_trace(go.Bar(
            x=[d['tag'] for d in tag_data],
            y=[d['accuracy'] for d in tag_data],
            name='准确率',
            marker_color='#2ecc71',
            width=0.5  # 设置条形宽度
        ))
        
        # 添加样本量散点图
        fig_accuracy.add_trace(go.Scatter(
            x=[d['tag'] for d in tag_data],
            y=[d['accuracy_total'] for d in tag_data],
            name='样本量',
            yaxis='y2',
            mode='markers+text',
            marker=dict(size=12, color='#3498db'),
            text=[d['accuracy_total'] for d in tag_data],
            textposition='top center'
        ))
        
        # 更新准确率图表布局
        fig_accuracy.update_layout(
            title={
                'text': '标签准确率分析',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis=dict(
                title='标签类型',
                tickangle=45
            ),
            yaxis=dict(
                title='准确率 (%)',
                side='left',
                range=[0, 100],
                gridcolor='lightgray'
            ),
            yaxis2=dict(
                title='样本量',
                side='right',
                overlaying='y',
                showgrid=False
            ),
            showlegend=True,
            legend={
                'orientation': 'h',
                'yanchor': 'bottom',
                'y': -0.2,
                'xanchor': 'center',
                'x': 0.5
            },
            height=800,
            margin=dict(t=100, b=150, l=60, r=60),
            plot_bgcolor='white',
            bargap=0.3  # 调整条形图间距
        )
        
        # 创建召回率分析图表
        fig_recall = go.Figure()
        
        # 添加召回率条形图
        fig_recall.add_trace(go.Bar(
            x=[d['tag'] for d in tag_data],
            y=[d['recall_rate'] for d in tag_data],
            name='召回率',
            marker_color='#f1c40f',
            width=0.5
        ))
        
        # 添加样本量散点图
        fig_recall.add_trace(go.Scatter(
            x=[d['tag'] for d in tag_data],
            y=[d['recall_total'] for d in tag_data],
            name='样本量',
            yaxis='y2',
            mode='markers+text',
            marker=dict(size=12, color='#3498db'),
            text=[d['recall_total'] for d in tag_data],
            textposition='top center'
        ))
        
        # 更新召回率图表布局
        fig_recall.update_layout(
            title={
                'text': '标签召回率分析',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis=dict(
                title='标签类型',
                tickangle=45
            ),
            yaxis=dict(
                title='召回率 (%)',
                side='left',
                range=[0, 100],
                gridcolor='lightgray'
            ),
            yaxis2=dict(
                title='样本量',
                side='right',
                overlaying='y',
                showgrid=False
            ),
            showlegend=True,
            legend=dict(
                orientation='h',      # 水平布局
                yanchor='bottom',    # 固定在底部
                y=-0.3,             # 向下移动位置，给两行图例留出空间
                xanchor='center',    # 居中对齐
                x=0.5,              # 居中位置
                traceorder='normal', # 保持图例顺序
                itemwidth=50,        # 设置图例项的宽度
                itemsizing='constant'# 保持图例项大小一致
            ),
            height=800,
            margin=dict(
                t=100,   # 顶部边距
                b=180,   # 增加底部边距，为两行图例留出空间
                l=60,    # 左边距
                r=60     # 右边距
            ),
            plot_bgcolor='white',
            bargap=0.3  # 调整条形图间距
        )
        
        # 保存图表
        fig_accuracy.write_html(f"{output_path}/tag_accuracy.html")
        fig_recall.write_html(f"{output_path}/tag_recall.html")
        
        # 添加标签格式处理函数
        def clean_tag_format(tag):
            """清理标签格式，去除前缀"""
            if ': ' in tag:
                return tag.split(': ')[1]
            return tag
        
        # 生成详细的标签分析报告
        tag_analysis = []
        for data in tag_data:
            tag_analysis.append({
                '标签': clean_tag_format(data['tag']),
                '样本量': data['recall_total'],
                '准确率': f"{data['accuracy']:.1f}%",
                '召回率': f"{data['recall_rate']:.1f}%",
                '正确数': data['correct'],
                '错误数': data['wrong'],
                '漏打数': data['missed']
            })
        
        # 更新报告内容
        report = {
            'total_statistics': total_stats,
            'tag_statistics': tag_stats,
            'tag_analysis': tag_analysis
        }
        
        with open(f"{output_path}/statistics_report.json", 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        return report
        
    except Exception as e:
        print(f"生成报告时发生错误: {str(e)}")
        return {
            'total_statistics': {
                'total_videos': len(records),
                'total_tags': 0,
                'correct_tags': 0,
                'wrong_tags': 0,
                'missed_tags': 0
            },
            'tag_statistics': {},
            'tag_analysis': []
        }

if __name__ == "__main__":
    jsonl_path = "./evaluation_data/evaluation_records.jsonl"
    output_path = "./reports"
    
    # 确保输出目录存在
    import os
    os.makedirs(output_path, exist_ok=True)
    
    # 生成报告
    report = generate_evaluation_report(jsonl_path, output_path)
    
    # 打印总体统计
    print("\n总体统计:")
    print(f"总视频数: {report['total_statistics']['total_videos']}")
    print(f"总标签数: {report['total_statistics']['total_tags']}")
    print(f"正确标签数: {report['total_statistics']['correct_tags']}")
    print(f"错误标签数: {report['total_statistics']['wrong_tags']}")
    print(f"遗漏标签数: {report['total_statistics']['missed_tags']}")
    print(f"总准确率: {(report['total_statistics']['correct_tags']/report['total_statistics']['total_tags']*100):.1f}%") 