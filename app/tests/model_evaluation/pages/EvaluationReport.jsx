import React, { useEffect, useState } from 'react';
import EvaluationCharts from '../components/EvaluationCharts';

const EvaluationReport = () => {
  const [reportData, setReportData] = useState(null);

  useEffect(() => {
    // 加载报告数据
    fetch('/api/evaluation-report')
      .then(res => res.json())
      .then(data => setReportData(data))
      .catch(err => console.error('加载评测报告失败:', err));
  }, []);

  if (!reportData) {
    return <div>加载中...</div>;
  }

  return (
    <div className="evaluation-report">
      <h1>模型评测报告</h1>
      <EvaluationCharts data={reportData} />
    </div>
  );
};

export default EvaluationReport; 