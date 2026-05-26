"""
Root Agent Node for ReAct LangGraph pipeline.

Root agent coordinates category analysis by spawning sub-agents,
reviewing results, and making strategic decisions about what to analyze next.
"""

import json
import logging
from pathlib import Path
from typing import Any

from tools import create_root_agent_registry, create_sub_agent_registry
from tools.registry import ToolResult
from prompts import build_react_system_prompt, load_skill

logger = logging.getLogger(__name__)

ROOT_TASK_PROMPT = """请对这段行车记录仪视频进行完整的多维度交通场景分析。

你需要依次分析以下10个类别，并通过 spawn_sub_agent 派出子agent完成每个类别的分析。

简单类别（通常不需要工具调用，可直接从视频判断）：
- 自然时间段、人工光源辅助、气象条件、路面状态与视线干扰、主干道路级别、特殊路段与设施、静态障碍与全局异常

复杂类别（需要视频切片等工具辅助分析）：
- 弱势交通参与者异常、车辆轨迹与空间冲突、极端高危与失控事件

策略建议：
1. 先用 get_video_info 了解视频基本信息
2. 先派简单类别（可以用 categories 参数批量派出）
3. 等简单类别完成后，把结果作为 hint 传给复杂类别
4. 用 review_result 检查结果质量
5. 所有类别完成后，确认分析结束
"""


async def root_agent_node(state: dict) -> dict:
    """
    Root Agent: coordinates all category analysis via sub-agents.

    Uses a ReAct loop with spawn_sub_agent, query_result, review_result tools.
    """
    client = state["qwen_client"]
    config = state.get("config", {})
    skills_dir = state.get("skills_dir", "skills")
    video_url = state.get("video_url", "")

    # Create tool registries
    sub_tool_registry = create_sub_agent_registry()
    root_tool_registry = create_root_agent_registry()

    # Build shared context for tool execution
    video_path = state.get("video_path", "")
    video_root = config.get("paths", {}).get("video_root", "")
    full_video_path = str(Path(video_root) / video_path) if video_path else ""

    context = {
        "video_path": full_video_path,
        "video_url": video_url,
        "video_info": state.get("video_info", {}),
        "output_dir": state.get("output_dir", ""),
        "config": config,
        "skills_dir": skills_dir,
        "qwen_client": client,
        "sub_tool_registry": sub_tool_registry,
        "completed_results": {},
        "on_sub_step": state.get("on_step"),  # Pass through to sub-agents
    }

    # Build root agent system prompt
    system_prompt = build_react_system_prompt(skills_dir)

    # Initialize system prompt on client
    client.init_system_prompt(system_prompt)

    # Run root agent ReAct loop
    max_turns = int(config.get("react_agent", {}).get("root_max_turns", 20))
    on_step = state.get("on_step")

    if on_step:
        try:
            on_step("start", {"message": "开始根Agent分析循环"})
        except Exception:
            pass

    content, conversation, trace = await client.run_react_session(
        video_url=video_url,
        task_description=ROOT_TASK_PROMPT,
        tool_registry=root_tool_registry,
        available_tool_names=["spawn_sub_agent", "query_category_result",
                               "get_video_info", "review_result"],
        context=context,
        max_turns=max_turns,
        on_step=on_step,
    )

    # Extract category results from context (populated by spawn_sub_agent)
    category_results = context.get("completed_results", {})

    if on_step:
        try:
            on_step("complete", {
                "message": f"分析完成，共 {len(category_results)} 个类别",
                "results": {cat: res.get("pred", []) for cat, res in category_results.items()},
            })
        except Exception:
            pass

    # Convert to legacy format for output_formatter compatibility
    merged_simple, complex_preds, simple_traces, complex_traces, agent_outputs = (
        _convert_to_legacy_format(category_results, state)
    )

    return {
        "category_results": category_results,
        "merged_simple": merged_simple,
        "complex_preds": complex_preds,
        "simple_traces": simple_traces,
        "complex_traces": complex_traces,
        "agent_outputs": agent_outputs,
    }


def _convert_to_legacy_format(
    category_results: dict,
    state: dict,
) -> tuple[dict, list[dict], list[dict], list[dict], list[dict]]:
    """Convert category_results to the format expected by output_formatter_node.

    Returns (merged_simple, complex_preds, simple_traces, complex_traces, agent_outputs).
    """
    config = state.get("config", {})
    video_info = state.get("video_info", {})
    duration = float(video_info.get("duration", 0) or 0)

    simple_categories = [
        "自然时间段", "人工光源辅助", "气象条件", "路面状态与视线干扰",
        "主干道路级别", "特殊路段与设施", "静态障碍与全局异常",
    ]
    complex_categories = [
        "弱势交通参与者异常", "车辆轨迹与空间冲突", "极端高危与失控事件",
    ]

    # Build merged_simple
    merged_simple = {}
    simple_traces = []
    for cat in simple_categories:
        result = category_results.get(cat)
        if result and isinstance(result, dict):
            merged_simple[cat] = result
            simple_traces.append({
                "category": cat,
                "raw_content": json.dumps(result, ensure_ascii=False),
                "trace": {"mode": "react_sub_agent"},
            })

    # Build complex_preds
    complex_preds = []
    complex_traces = []
    for cat in complex_categories:
        result = category_results.get(cat)
        if result and isinstance(result, dict):
            complex_preds.append({
                "category": cat,
                "sub_agent_id": f"react_{cat}",
                "time_slice": {"start": 0, "end": duration, "reason": "ReAct全视频分析"},
                "pred": result.get("pred", []),
                "events": result.get("events", []),
                "step1_object_detection": result.get("step1_object_detection", ""),
                "step2_motion_analysis": result.get("step2_motion_analysis", ""),
                "step3_conflict_check": result.get("step3_conflict_check", ""),
            })
            complex_traces.append({
                "category": cat,
                "raw_content": json.dumps(result, ensure_ascii=False),
                "trace": {"mode": "react_sub_agent"},
            })

    # Build agent_outputs
    agent_outputs = [
        {
            "agent_name": "root_agent_react",
            "task_name": "video_analysis_root",
            "raw_content": "react_root_agent",
            "result": {
                "categories_completed": list(category_results.keys()),
            },
            "trace": {"mode": "react"},
        }
    ]

    return merged_simple, complex_preds, simple_traces, complex_traces, agent_outputs
