# clip_select 工具（局部补充模式）

你当前收到的已经是预切好的局部时间窗视频，**时间轴从 0 秒开始**。

若该切片中证据不足，可用 clip_select 对本切片内的局部时段做更细粒度采样：

<tool>
{"tool":"clip_select","arguments":{"start_time":0,"end_time":4}}
</tool>

参数：start_time(float, ≥0)、end_time(float, >start_time)。sampling_interval 由配置注入，不要填写。

## 约束

- 时间使用局部相对时间（0 ~ {slice_duration} 秒），不要使用原始全视频的绝对时间
- 证据充分则直接输出 JSON，不必强制调用工具
- 最多补切 2 次；仍不足则以已有信息作出最佳判断
- start_time/end_time 只保留一位小数（如 3.0）
