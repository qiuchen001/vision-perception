# yolo_detect 工具

对指定时间段运行 YOLO 目标检测，快速识别行人、车辆、自行车等目标。

## 何时使用

- 需要快速确认某区域是否存在特定目标
- 需要统计某段时间内的目标数量和分布
- 作为 clip_select 的补充，先用 YOLO 定位再用切片仔细观察

## 注意

- 此工具仅返回目标检测结果，不做行为分析
- 需要分析目标行为时，应使用 clip_select 提取切片仔细观察

## 参数

- start_time (number, 必需): 检测起始时间（秒）
- end_time (number, 必需): 检测结束时间（秒）
- conf_threshold (number, 默认0.35): 置信度阈值

## 调用格式

<tool>
{"tool":"yolo_detect","arguments":{"start_time":5,"end_time":20,"conf_threshold":0.35}}
</tool>
