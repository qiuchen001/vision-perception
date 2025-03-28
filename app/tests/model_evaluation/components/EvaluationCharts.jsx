import React from 'react';
import { Pie, Bar } from '@ant-design/plots';

const EvaluationCharts = ({ data }) => {
  const { total_statistics: totalStats, tag_statistics: tagStats } = data;

  // 准备饼图数据
  const pieData = [
    { type: '正确标签', value: totalStats.correct_tags },
    { type: '错误标签', value: totalStats.wrong_tags },
    { type: '遗漏标签', value: totalStats.missed_tags },
  ];

  // 准备条形图数据
  const barData = Object.entries(tagStats).map(([tag, stats]) => ({
    tag,
    accuracy: (stats.correct / stats.total * 100).toFixed(1),
    total: stats.total,
    correct: stats.correct,
    wrong: stats.wrong,
    missed: stats.missed,
  })).sort((a, b) => b.accuracy - a.accuracy);

  // 饼图配置
  const pieConfig = {
    data: pieData,
    angleField: 'value',
    colorField: 'type',
    radius: 0.8,
    innerRadius: 0.6,
    label: {
      type: 'inner',
      offset: '-30%',
      content: ({ percent }) => `${(percent * 100).toFixed(1)}%`,
      style: {
        fontSize: 14,
        textAlign: 'center',
      },
    },
    interactions: [{ type: 'element-active' }],
    statistic: {
      title: {
        content: '总准确率',
        style: {
          fontSize: '16px',
        },
      },
      content: {
        content: `${(totalStats.correct_tags / totalStats.total_tags * 100).toFixed(1)}%`,
        style: {
          fontSize: '24px',
        },
      },
    },
  };

  // 条形图配置
  const barConfig = {
    data: barData,
    xField: 'accuracy',
    yField: 'tag',
    seriesField: 'tag',
    meta: {
      accuracy: {
        formatter: (v) => `${v}%`,
      },
    },
    label: {
      position: 'right',
      formatter: (v) => `${v.accuracy}%`,
    },
    tooltip: {
      fields: ['tag', 'accuracy', 'total', 'correct', 'wrong', 'missed'],
      formatter: (datum) => {
        return [
          { name: '标签', value: datum.tag },
          { name: '准确率', value: `${datum.accuracy}%` },
          { name: '总数', value: datum.total },
          { name: '正确', value: datum.correct },
          { name: '错误', value: datum.wrong },
          { name: '遗漏', value: datum.missed },
        ];
      },
    },
  };

  return (
    <div className="evaluation-charts">
      <div className="statistics-summary" style={{ marginBottom: '20px' }}>
        <h2>评测统计概览</h2>
        <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap' }}>
          <div>总视频数: {totalStats.total_videos}</div>
          <div>总标签数: {totalStats.total_tags}</div>
          <div>正确标签: {totalStats.correct_tags}</div>
          <div>错误标签: {totalStats.wrong_tags}</div>
          <div>遗漏标签: {totalStats.missed_tags}</div>
        </div>
      </div>

      <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap' }}>
        <div style={{ flex: 1, minWidth: '300px' }}>
          <h3>标签评测结果分布</h3>
          <Pie {...pieConfig} />
        </div>
        <div style={{ flex: 2, minWidth: '500px' }}>
          <h3>各类标签准确率</h3>
          <Bar {...barConfig} />
        </div>
      </div>
    </div>
  );
};

export default EvaluationCharts; 