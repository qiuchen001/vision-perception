"""
spawn_sub_agent tool — Root Agent dispatches sub-agents for category analysis.
"""

import json
import logging

from tools.registry import ToolDefinition, ToolResult

logger = logging.getLogger(__name__)

ALL_CATEGORIES = [
    "自然时间段", "人工光源辅助", "气象条件", "路面状态与视线干扰",
    "主干道路级别", "特殊路段与设施", "静态障碍与全局异常",
    "弱势交通参与者异常", "车辆轨迹与空间冲突", "极端高危与失控事件",
]

COMPLEX_CATEGORIES = {"弱势交通参与者异常", "车辆轨迹与空间冲突", "极端高危与失控事件"}


async def execute_spawn_sub_agent(args: dict, context: dict) -> ToolResult:
    """Execute spawn_sub_agent: create a sub-agent to analyze a category."""
    category = args.get("category", "")
    hint = args.get("hint", "")
    categories_list = args.get("categories")  # Support batch spawn

    # Handle batch spawn
    if categories_list and isinstance(categories_list, list):
        return await _execute_batch_spawn(categories_list, hint, context)

    if not category or category not in ALL_CATEGORIES:
        return ToolResult(
            output=f"错误：未知类别「{category}」。可用类别：{', '.join(ALL_CATEGORIES)}"
        )

    # Check if already completed
    completed_results = context.get("completed_results", {})
    if category in completed_results and not args.get("force_redo", False):
        return ToolResult(
            output=f"类别「{category}」已完成分析，无需重复。使用 query_category_result 查看结果。"
        )

    # Run sub-agent
    try:
        result = await _run_sub_agent(category, hint, context)
        return result
    except Exception as e:
        logger.error("Sub-agent error for %s: %s", category, e)
        return ToolResult(output=f"子agent分析「{category}」时出错: {e}")


async def _execute_batch_spawn(
    categories: list[str],
    hint: str,
    context: dict,
) -> ToolResult:
    """Spawn multiple sub-agents in parallel."""
    import asyncio

    tasks = [_run_sub_agent(cat, hint, context) for cat in categories]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    lines = ["批量子agent分析结果："]
    for cat, result in zip(categories, results):
        if isinstance(result, Exception):
            lines.append(f"\n「{cat}」失败: {result}")
        else:
            # result is a ToolResult
            completed_results = context.get("completed_results", {})
            if cat in completed_results:
                pred = completed_results[cat].get("pred", [])
                lines.append(f"\n「{cat}」完成: pred={json.dumps(pred, ensure_ascii=False)}")
            else:
                lines.append(f"\n「{cat}」: {result.output[:100]}")

    return ToolResult(output="\n".join(lines))


async def _run_sub_agent(
    category: str,
    hint: str,
    context: dict,
) -> ToolResult:
    """Run a single sub-agent for one category."""
    from prompts import load_category_skill
    from response_parser import ResponseParser

    client = context.get("qwen_client")
    config = context.get("config", {})
    skills_dir = context.get("skills_dir", "skills")
    sub_tool_registry = context.get("sub_tool_registry")
    on_step = context.get("on_sub_step")

    # Build task description
    category_prompt = load_category_skill(category, skills_dir)
    if hint:
        category_prompt = f"上级agent提示：{hint}\n\n{category_prompt}"

    # Determine available tools
    is_complex = category in COMPLEX_CATEGORIES
    if is_complex:
        tool_names = ["clip_select", "frame_extract", "yolo_detect",
                      "query_category_result", "get_video_info"]
    else:
        tool_names = ["get_video_info", "query_category_result"]

    # Determine if abnormal
    from prompts import is_abnormal_category
    is_abnormal = is_abnormal_category(category, config)

    # Run ReAct session
    max_turns = int(config.get("react_agent", {}).get("sub_max_turns", 10))
    content, conversation, trace = await client.run_react_session(
        video_url=context["video_url"],
        task_description=category_prompt,
        tool_registry=sub_tool_registry,
        available_tool_names=tool_names,
        context={
            **context,
            "completed_results": context.get("completed_results", {}),
        },
        max_turns=max_turns,
        on_step=on_step,
    )

    # Parse result
    parsed = ResponseParser.parse(category, content, is_abnormal=is_abnormal)
    if ResponseParser.is_parse_failed(parsed):
        # Fallback
        from response_parser import ResponseParser as RP
        default_pred = RP.get_default_normal_pred(category)
        parsed = {
            "pred": default_pred if isinstance(default_pred, list) else [default_pred],
            "events": [],
            "_parse_failed": True,
        }

    # Store result in context (mutable, shared across calls)
    completed_results = context.setdefault("completed_results", {})
    completed_results[category] = parsed

    # Build summary for root agent
    pred = parsed.get("pred", [])
    events = parsed.get("events", [])
    events_summary = ""
    if events:
        events_summary = f"，{len(events)}个事件"
        for evt in events[:3]:
            events_summary += f"\n    - {evt.get('type', '?')}: {evt.get('start_time', '?')}s-{evt.get('end_time', '?')}s"

    return ToolResult(
        output=f"子agent完成「{category}」分析：pred={json.dumps(pred, ensure_ascii=False)}{events_summary}",
        metadata={
            "category": category,
            "parsed": parsed,
            "trace": trace,
            "conversation_length": len(conversation),
        },
    )


SPAWN_SUB_AGENT_TOOL = ToolDefinition(
    name="spawn_sub_agent",
    description=(
        "派出一个子agent分析指定类别。子agent拥有自己的思考→行动→观察循环，"
        "可以自主使用工具（视频切片、帧提取、目标检测等）完成分析。"
        "复杂类别（弱势交通参与者异常、车辆轨迹与空间冲突、极端高危与失控事件）"
        "的子agent可以使用全部分析工具；简单类别只使用基础工具。"
        "可以通过 hint 参数给子agent传递上下文信息。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "description": "要分析的类别名",
                "enum": ALL_CATEGORIES,
            },
            "hint": {
                "type": "string",
                "description": "给子agent的提示信息（如已知的上下文、重点关注方向）",
            },
            "categories": {
                "type": "array",
                "items": {"type": "string"},
                "description": "批量模式：同时分析多个类别（并行执行）",
            },
            "force_redo": {
                "type": "boolean",
                "description": "是否强制重新分析已完成的类别",
                "default": False,
            },
        },
        "required": ["category"],
    },
    executor=execute_spawn_sub_agent,
)
