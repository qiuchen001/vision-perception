"""
query_category_result tool — look up results from already-completed categories.

Enables cross-category reasoning: the agent can check if it found "雨天"
in weather before concluding about road conditions.
"""

import json
import logging

from tools.registry import ToolDefinition, ToolResult

logger = logging.getLogger(__name__)


async def execute_query_category_result(args: dict, context: dict) -> ToolResult:
    """Execute query_category_result: look up completed category results."""
    category = args.get("category", "")
    if not category:
        return ToolResult(output="错误：未提供类别名称")

    completed_results = context.get("completed_results", {})
    result = completed_results.get(category)

    if result is None:
        available = list(completed_results.keys())
        available_str = "、".join(available) if available else "（无）"
        return ToolResult(
            output=f"类别「{category}」尚未完成分析。已完成类别：{available_str}"
        )

    # Format the result
    if isinstance(result, dict):
        output_lines = [f"类别「{category}」分析结果："]
        pred = result.get("pred", [])
        if pred:
            output_lines.append(f"  pred: {json.dumps(pred, ensure_ascii=False)}")
        events = result.get("events", [])
        if events:
            output_lines.append(f"  events ({len(events)}个):")
            for evt in events[:5]:
                evt_type = evt.get("type", "?")
                start = evt.get("start_time", "?")
                end = evt.get("end_time", "?")
                output_lines.append(f"    - {evt_type}: {start}s-{end}s")
            if len(events) > 5:
                output_lines.append(f"    ... 还有 {len(events) - 5} 个事件")
        # Include step summaries
        for step_key in ["step0_environment_context", "evidence",
                         "step1_object_detection", "align",
                         "step2_motion_analysis", "decision",
                         "step3_conflict_check"]:
            step_val = result.get(step_key, "")
            if step_val:
                output_lines.append(f"  {step_key}: {str(step_val)[:200]}")

        return ToolResult(
            output="\n".join(output_lines),
            metadata={"category": category, "result": result},
        )
    else:
        return ToolResult(
            output=f"类别「{category}」结果: {json.dumps(result, ensure_ascii=False)}"
        )


QUERY_CATEGORY_RESULT_TOOL = ToolDefinition(
    name="query_category_result",
    description=(
        "查询已完成的类别分析结果。可用于跨类别推理，例如："
        "查到「气象条件」为雨天后再分析路面状态时考虑积水因素。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "description": "要查询的类别名称，如「气象条件」「自然时间段」等",
            },
        },
        "required": ["category"],
    },
    executor=execute_query_category_result,
)
