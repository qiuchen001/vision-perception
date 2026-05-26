"""
review_result tool — Root Agent reviews sub-agent results for quality.
"""

import json
import logging

from tools.registry import ToolDefinition, ToolResult

logger = logging.getLogger(__name__)

COMPLEX_CATEGORIES = {"弱势交通参与者异常", "车辆轨迹与空间冲突", "极端高危与失控事件"}

# Forbidden pred values
FORBIDDEN_PREDS = {"其他异常情况"}


async def execute_review_result(args: dict, context: dict) -> ToolResult:
    """Execute review_result: verify a category result for quality issues."""
    category = args.get("category", "")
    completed_results = context.get("completed_results", {})
    video_info = context.get("video_info", {})
    duration = float(video_info.get("duration", 0) or 0)

    if not category:
        # Review all completed results
        all_issues = {}
        for cat, result in completed_results.items():
            issues = _check_result(cat, result, duration)
            if issues:
                all_issues[cat] = issues
        if all_issues:
            lines = ["校验发现以下问题："]
            for cat, issues in all_issues.items():
                lines.append(f"\n「{cat}」：")
                for issue in issues:
                    lines.append(f"  - {issue}")
            return ToolResult(output="\n".join(lines), metadata={"issues": all_issues})
        else:
            return ToolResult(output="所有已完成类别的结果校验通过，无问题。")

    # Review specific category
    result = completed_results.get(category)
    if result is None:
        return ToolResult(output=f"类别「{category}」尚未完成分析，无法审查。")

    issues = _check_result(category, result, duration)
    if not issues:
        return ToolResult(output=f"类别「{category}」结果校验通过，无问题。")

    lines = [f"类别「{category}」校验发现 {len(issues)} 个问题："]
    for issue in issues:
        lines.append(f"  - {issue}")
    lines.append("\n建议：使用 spawn_sub_agent 重新分析该类别（设置 force_redo=true）。")

    return ToolResult(
        output="\n".join(lines),
        metadata={"category": category, "issues": issues},
    )


def _check_result(category: str, result: dict, duration: float) -> list[str]:
    """Programmatic verification of a category result."""
    if not isinstance(result, dict):
        return ["结果格式错误：应为字典"]

    issues = []

    # Check required fields
    is_complex = category in COMPLEX_CATEGORIES
    if "pred" not in result:
        issues.append("缺少 pred 字段")
    elif not isinstance(result.get("pred"), list):
        issues.append("pred 应为数组")
    elif not result["pred"]:
        issues.append("pred 为空")

    if is_complex and "events" not in result:
        issues.append("复杂类别缺少 events 字段")

    # Check forbidden values
    pred = result.get("pred", [])
    for p in pred:
        if p in FORBIDDEN_PREDS:
            issues.append(f"pred 包含禁止值「{p}」，需改写为具体描述")

    # Check event time bounds
    events = result.get("events", [])
    if isinstance(events, list):
        for i, evt in enumerate(events):
            if not isinstance(evt, dict):
                continue
            start = evt.get("start_time", 0)
            end = evt.get("end_time", 0)
            if isinstance(start, (int, float)) and start < 0:
                issues.append(f"events[{i}] start_time < 0")
            if isinstance(end, (int, float)) and duration > 0 and end > duration + 1:
                issues.append(f"events[{i}] end_time ({end:.1f}s) 超出视频时长 ({duration:.1f}s)")
            if isinstance(start, (int, float)) and isinstance(end, (int, float)) and end < start:
                issues.append(f"events[{i}] end_time < start_time")

    # Check event type matches pred
    if isinstance(events, list) and isinstance(pred, list):
        pred_set = set(str(p) for p in pred)
        for i, evt in enumerate(events):
            if isinstance(evt, dict):
                evt_type = str(evt.get("type", ""))
                if evt_type and evt_type not in pred_set:
                    issues.append(f"events[{i}] type「{evt_type}」不在 pred 中")

    # Check step fields for complex categories
    if is_complex:
        for step_key in ["step0_environment_context", "step1_object_detection",
                         "step2_motion_analysis", "step3_conflict_check"]:
            val = result.get(step_key, "")
            if not val or (isinstance(val, str) and not val.strip()):
                issues.append(f"步骤字段 {step_key} 为空")

    return issues


REVIEW_RESULT_TOOL = ToolDefinition(
    name="review_result",
    description=(
        "审查指定类别的分析结果质量。检查格式是否正确、事件时间是否合理、"
        "pred与events是否一致等。可以审查特定类别或所有已完成类别。"
        "发现问题后建议使用 spawn_sub_agent 重新分析。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "description": "要审查的类别名称（不填则审查所有已完成类别）",
            },
        },
        "required": [],
    },
    executor=execute_review_result,
)
