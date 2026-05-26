# query_category_result 工具

查询已完成的类别分析结果，用于跨类别推理。

## 何时使用

- 复杂类别分析时，需要参考简单类别的结果
- 例如：查到「气象条件」为雨天后再分析路面状态时考虑积水因素
- 例如：查到「自然时间段」为夜晚后再分析弱势参与者时注意低照度

## 参数

- category (string, 必需): 要查询的类别名称

## 调用格式

<tool>
{"tool":"query_category_result","arguments":{"category":"气象条件"}}
</tool>
