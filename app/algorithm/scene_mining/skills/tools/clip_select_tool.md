# clip_select 工具

提取视频指定时间段的切片，用于仔细观察可疑区域。

## 何时使用

- 首次观察视频时，用 overview 级别概览全片或大段
- 发现可疑时间窗，用 scan 级别验证假设
- 确认异常存在，用 focus 级别聚焦关键段
- 需要精确定位事件边界，用 pinpoint 级别

## 渐进采样策略

| 级别 | FPS | 窗口时长 | 适用场景 |
|------|-----|---------|---------|
| overview | 1 | 30s | 首次概览，形成假设 |
| scan | 4 | 10s | 扫描可疑区域，验证假设 |
| focus | 8 | 6s | 聚焦关键段，观察行为 |
| pinpoint | 16 | 2s | 精确定位事件边界 |

## 参数

- start_time (number, 必需): 起始时间（秒），≥0
- end_time (number, 必需): 结束时间（秒），>start_time
- sampling_level (string, 默认"scan"): 采样精度级别

## 调用格式

<tool>
{"tool":"clip_select","arguments":{"start_time":5,"end_time":15,"sampling_level":"scan"}}
</tool>
