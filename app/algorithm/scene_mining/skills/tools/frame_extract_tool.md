# frame_extract 工具

提取指定时间点的高分辨率视频帧，用于仔细检查特定时刻的细节。

## 何时使用

- 需要看清目标外观（如车辆型号、行人服装）
- 需要读取路标、信号灯、文字等静态信息
- 需要对比不同时间点的画面变化

## 参数

- timestamps (array of numbers, 必需): 要提取的时间点列表（秒），如 [3.5, 7.2, 12.0]，最多5个

## 调用格式

<tool>
{"tool":"frame_extract","arguments":{"timestamps":[5.0,10.0,15.0]}}
</tool>
