## 分析步骤

1) 提炼关键视觉证据（目标、动作、环境、时间线）
2) 比较候选子类与视觉证据的一致性
3) 输出最符合的结果

## JSON Schema

{
  "evidence": "关键视觉证据（string，1-2句）",
  "align": "候选子类与证据的一致性比较（string，1-2句）",
  "decision": "最终结论与排除理由（string，1-2句）",
  "pred": "子类中文名数组（list[string]），无法判断时返回 []"
}

## 约束

- JSON 顺序：evidence -> align -> decision -> pred
- 所有字段必须非空