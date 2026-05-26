# clip_select 工具

当需要核验细节时，使用 clip_select 获取指定时间段视频切片。

## 调用格式

必须使用 <tool></tool> 包裹：

<tool>
{"tool":"clip_select","arguments":{"start_time":0,"end_time":10}}
</tool>

参数：start_time(float, ≥0)、end_time(float, >start_time)。sampling_interval 由配置注入，不要填写。

## 采样策略

clip_select 仅做局部补充，不再全视频扫描：
1) 首次：覆盖可疑时间窗，验证候选异常
2) 二次（可选）：缩小到事件边界
3) 三次（可选）：精确定位
4) 证据充分再输出 JSON，不足则继续补切片

## 时间约束

- 输出 events 前先用局部切片核验时间边界
- 无可确认事件时 events 返回 []
- start_time/end_time 只保留一位小数（如 15.0）