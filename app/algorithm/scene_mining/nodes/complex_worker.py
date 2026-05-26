"""
Complex Worker Node for LangGraph pipeline.

Processes complex categories with multi-round tool calls.
Each worker handles one category and can loop internally with tool calls.
"""

import logging
import json
from pathlib import Path
from typing import Any

from response_parser import ResponseParser

logger = logging.getLogger(__name__)


def _apply_offset_to_events(events: list[Any], offset: float) -> list[dict]:
    adjusted: list[dict] = []
    for item in events:
        if not isinstance(item, dict):
            continue
        evt = dict(item)
        try:
            evt["start_time"] = float(evt.get("start_time", 0.0)) + offset
            evt["end_time"] = float(evt.get("end_time", 0.0)) + offset
        except (TypeError, ValueError):
            pass
        adjusted.append(evt)
    return adjusted


def _sanitize_local_slice_events(events: list[Any], slice_duration: float, epsilon: float = 0.15) -> tuple[list[dict], dict]:
    """Sanitize events in local slice timeline before global offset mapping.

    Rules:
    - drop if start_time is outside local slice [-epsilon, slice_duration + epsilon]
    - clip start_time/end_time to [0, slice_duration] if within epsilon tolerance
    """
    duration = max(0.0, float(slice_duration or 0.0))
    kept: list[dict] = []
    dropped_count = 0
    clipped_count = 0
    swapped_count = 0
    invalid_count = 0

    for item in events:
        if not isinstance(item, dict):
            invalid_count += 1
            continue
        evt = dict(item)
        try:
            start = float(evt.get("start_time", 0.0))
            end = float(evt.get("end_time", start))
        except (TypeError, ValueError):
            invalid_count += 1
            continue

        if end < start:
            start, end = end, start
            swapped_count += 1

        # start out of local clip bounds (with epsilon tolerance) -> drop
        if start < -epsilon or start > duration + epsilon:
            dropped_count += 1
            continue

        # clamp to valid range
        if start < 0.0:
            start = 0.0
            clipped_count += 1
        if start > duration:
            start = duration
            clipped_count += 1

        # start is valid but end exceeds -> clip
        if end > duration:
            end = duration
            clipped_count += 1

        # ensure non-negative span
        if end < start:
            end = start

        evt["start_time"] = start
        evt["end_time"] = end
        kept.append(evt)

    stats = {
        "input_count": len(events) if isinstance(events, list) else 0,
        "kept_count": len(kept),
        "dropped_count": dropped_count,
        "clipped_count": clipped_count,
        "swapped_count": swapped_count,
        "invalid_count": invalid_count,
        "slice_duration": duration,
    }
    return kept, stats


def _extract_json_object(text: str) -> dict | None:
    content = (text or "").strip()
    for start in range(len(content)):
        if content[start] != "{":
            continue
        depth = 0
        for idx in range(start, len(content)):
            char = content[idx]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(content[start:idx + 1])
                except json.JSONDecodeError:
                    break
                if isinstance(obj, dict):
                    return obj
                break
    return None


def _inject_offset_into_trace(trace: dict, default_offset: float) -> dict:
    trace_obj = dict(trace) if isinstance(trace, dict) else {}
    rounds = trace_obj.get("rounds", [])
    if not isinstance(rounds, list):
        trace_obj["global_time_offset"] = default_offset
        return trace_obj
    for round_item in rounds:
        if not isinstance(round_item, dict):
            continue
        round_offset = round_item.get("global_time_offset", default_offset)
        try:
            round_offset = float(round_offset)
        except (TypeError, ValueError):
            round_offset = default_offset
        tool_calls = round_item.get("tool_calls", [])
        if isinstance(tool_calls, list):
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                normalized = call.get("normalized")
                if isinstance(normalized, dict):
                    try:
                        call["global_normalized"] = {
                            **normalized,
                            "start_time": float(normalized.get("start_time", 0.0)) + round_offset,
                            "end_time": float(normalized.get("end_time", 0.0)) + round_offset,
                        }
                    except (TypeError, ValueError):
                        pass
        content = round_item.get("content")
        if isinstance(content, str):
            obj = _extract_json_object(content)
            if isinstance(obj, dict):
                events = obj.get("events", [])
                if isinstance(events, list):
                    obj["events"] = _apply_offset_to_events(events, round_offset)
                    round_item["content_local"] = content
                    round_item["content"] = json.dumps(obj, ensure_ascii=False)
    trace_obj["global_time_offset"] = default_offset
    return trace_obj


async def complex_worker_node(state: dict) -> dict:
    """
    Complex worker node for processing complex categories with tool calls.

    Input state keys used:
    - task: dict with {"category": str, "time_slice": {"start": float, "end": float}, ...}
    - video_url: str
    - video_path: str
    - output_dir: str
    - qwen_client: QwenClient
    - config: dict
    - skills_dir: str
    - merged_simple: dict (for context)

    Output state keys set:
    - complex_pred: dict with {"category": str, "pred": list, "events": list}

    Note: This runs as a Send target with multi-round tool call capability.
    """
    task = state.get("task", {})
    category = task.get("category")

    if not category:
        logger.error("Complex worker: no category in task")
        return {"complex_pred": {"category": "unknown", "pred": [], "events": []}}

    client = state["qwen_client"]
    video_url = state["video_url"]
    video_path = state["video_path"]
    output_dir = state["output_dir"]
    config = state["config"]
    skills_dir = state["skills_dir"]
    on_step = state.get("on_step")
    merged_simple = state.get("merged_simple", {})
    time_slice = state.get("time_slice", {})
    slice_idx = int(state.get("slice_idx", 0) or 0)
    sub_agent_id = str(state.get("sub_agent_id", f"{category}_slice_1"))

    # Build previous preds context from merged simple results
    previous_preds = {}
    for cat, result in merged_simple.items():
        if isinstance(result, dict):
            previous_preds[cat] = result.get("pred", [])

    # Load category prompt; conditionally include tool call instructions
    from prompts import load_category_skill, build_complex_tool_prompt
    tool_call_enabled = config.get("tool_call", {}).get("complex_worker_tool_call_enabled", True)
    if tool_call_enabled:
        category_prompt = build_complex_tool_prompt(skills_dir) + "\n\n" + load_category_skill(category, skills_dir)
    else:
        category_prompt = load_category_skill(category, skills_dir)
    slice_start = float(time_slice.get("start", 0.0)) if isinstance(time_slice, dict) else 0.0
    slice_end = float(time_slice.get("end", 0.0)) if isinstance(time_slice, dict) else 0.0
    slice_reason = str(time_slice.get("reason", "") or "") if isinstance(time_slice, dict) else ""
    suspected_evidence = str(time_slice.get("suspected_evidence", "") or "") if isinstance(time_slice, dict) else str(task.get("suspected_evidence", "") or "")
    if on_step:
        on_step(
            "abnormal",
            f"异常事件分析中... {category}",
            {"category": category, "slice_start": slice_start, "slice_end": slice_end},
        )

    supervisor_hints: list[str] = []
    if suspected_evidence:
        supervisor_hints.append(f"规划阶段参考提示（仅供参考，你必须基于视觉证据独立判断）：{suspected_evidence}")
    supervisor_hint_block = "\n".join(supervisor_hints) if supervisor_hints else ""

    slice_prefix = (
        f"当前你是子Agent（{sub_agent_id}），负责优先精查时间窗 [{slice_start:.2f}s, {slice_end:.2f}s]。\n"
        f"时间窗选择依据：{slice_reason if slice_reason else '全局扫描'}。\n"
    )
    if supervisor_hint_block:
        slice_prefix += f"{supervisor_hint_block}\n"
    slice_prefix += (
        "注意：你当前收到的是该时间窗的局部切片视频，当前输入视频时间轴从 0 秒开始。\n"
    )
    if tool_call_enabled:
        slice_prefix += (
            f"若你继续调用 clip_select，请使用局部相对时间（0 ~ {max(0.1, slice_end - slice_start):.2f} 秒），"
            "不要再使用原始全视频的绝对时间。\n"
        )
    slice_prefix += "请重点围绕该时间窗判断；若证据不足可扩大，但必须在输出的步骤中说明。\n\n"
    category_prompt = slice_prefix + category_prompt
    slice_clip_info = {"clip_path": "", "saved_clip_path": ""}

    try:
        source_video_path = str((Path(client.config["paths"]["video_root"]) / video_path).resolve())
        interval = client._resolve_clip_sampling_interval(category, 0)
        slice_clip_info = await client._call_clip_select(
            source_video_path,
            {
                "start_time": slice_start,
                "end_time": slice_end,
                "sampling_interval": interval,
            },
            output_dir=output_dir,
            category_name=category,
            round_idx=0,
            call_idx=slice_idx,
            save_name_prefix=f"pre_slice_{sub_agent_id}",
        )
        sliced_video_url = f"file://{slice_clip_info['clip_path']}"
        local_slice_duration = max(0.1, slice_end - slice_start)
        base_category_prompt = category_prompt
        max_event_time_retries = config.get("retry", {}).get("max_event_time_retries", 1)
        event_time_retry_count = 0

        for time_retry in range(max_event_time_retries + 1):
            effective_prompt = base_category_prompt
            if time_retry > 0:
                correction = (
                    f"\n\n⚠️ 重要纠正：你上一次输出的 events 时间超出了当前切片范围，被系统丢弃。\n"
                    f"当前输入是局部切片视频，时间轴从 0 秒开始，总时长 {local_slice_duration:.2f} 秒。\n"
                    f"所有 events 必须满足：0 <= start_time <= end_time <= {local_slice_duration:.2f}。\n"
                    f"不要使用原始全视频的绝对时间。如果事件从切片开头就存在，请设 start_time=0；"
                    f"如果持续到切片结尾，请设 end_time={local_slice_duration:.2f}。\n"
                )
                effective_prompt = base_category_prompt + correction
                logger.warning(
                    "[%s] Event time mismatch detected, retrying with correction (%d/%d)",
                    category, time_retry, max_event_time_retries,
                )

            _, raw_content, result, trace = await client.analyze_category_sequential(
                video_url=sliced_video_url,
                category_name=category,
                category_prompt=effective_prompt,
                previous_preds=previous_preds,
                output_dir=output_dir,
                is_abnormal=True,
                detailed=False,
                pre_sliced_input=True,
                sampling_round_offset=1,
                global_time_offset=slice_start,
                video_duration=local_slice_duration,
            )

            pred = result.get("pred", []) if isinstance(result, dict) else []
            raw_events = result.get("events", []) if isinstance(result, dict) else []
            sanitized_events, sanitize_stats = _sanitize_local_slice_events(
                raw_events if isinstance(raw_events, list) else [],
                local_slice_duration,
            )

            # Only retry if ALL events were dropped (systematic timeline error)
            all_dropped = (
                sanitize_stats["input_count"] > 0
                and sanitize_stats["kept_count"] == 0
                and sanitize_stats["dropped_count"] == sanitize_stats["input_count"]
            )
            if not all_dropped or time_retry >= max_event_time_retries:
                event_time_retry_count = time_retry
                break

        final_offset = trace.get("final_global_offset", slice_start) if isinstance(trace, dict) else slice_start
        try:
            final_offset = float(final_offset)
        except (TypeError, ValueError):
            final_offset = slice_start
        events = _apply_offset_to_events(sanitized_events, final_offset)
        step1_text = result.get("step1_object_detection", "") if isinstance(result, dict) else ""
        step2_text = result.get("step2_motion_analysis", "") if isinstance(result, dict) else ""
        step3_text = result.get("step3_conflict_check", "") if isinstance(result, dict) else ""
        trace = _inject_offset_into_trace(trace if isinstance(trace, dict) else {}, slice_start)
        trace["local_event_sanitization"] = sanitize_stats
        if event_time_retry_count > 0:
            trace["event_time_retry_count"] = event_time_retry_count
            trace["event_time_retry_exhausted"] = (
                sanitize_stats["kept_count"] == 0 and sanitize_stats["dropped_count"] > 0
            )

        logger.info("Complex worker completed: %s -> pred=%s, events=%d",
                    category, pred[:2] if pred else [], len(events))

        return {
            "complex_preds": [
                {
                    "category": category,
                    "sub_agent_id": sub_agent_id,
                    "time_slice": {"start": slice_start, "end": slice_end, "reason": slice_reason},
                    "pred": pred,
                    "events": events,
                    "step1_object_detection": step1_text,
                    "step2_motion_analysis": step2_text,
                    "step3_conflict_check": step3_text,
                }
            ],
            "complex_traces": [{"category": category, "raw_content": raw_content, "trace": trace}],
            "agent_outputs": [
                {
                    "agent_name": f"complex_worker_{category}_{sub_agent_id}",
                    "task_name": f"{category} [{slice_start:.2f}s-{slice_end:.2f}s]",
                    "raw_content": raw_content,
                    "result": {
                        "pred": pred,
                        "events": events,
                        "time_slice": {"start": slice_start, "end": slice_end, "reason": slice_reason},
                        "pre_sliced_clip_path": slice_clip_info.get("saved_clip_path", ""),
                    },
                    "trace": trace,
                }
            ],
        }

    except Exception as e:
        logger.error("Complex worker error for %s: %s", category, e)
        default_pred = ResponseParser.get_default_normal_pred(category)
        return {
            "complex_preds": [
                {
                    "category": category,
                    "sub_agent_id": sub_agent_id,
                    "time_slice": {"start": slice_start, "end": slice_end, "reason": slice_reason},
                    "pred": default_pred if isinstance(default_pred, list) else [default_pred],
                    "events": [],
                    "step1_object_detection": "",
                    "step2_motion_analysis": "",
                    "step3_conflict_check": "",
                }
            ],
            "complex_traces": [{"category": category, "raw_content": "", "trace": {}}],
            "agent_outputs": [
                {
                    "agent_name": f"complex_worker_{category}_{sub_agent_id}",
                    "task_name": f"{category} [{slice_start:.2f}s-{slice_end:.2f}s]",
                    "raw_content": "",
                    "result": {
                        "pred": default_pred if isinstance(default_pred, list) else [default_pred],
                        "events": [],
                        "time_slice": {"start": slice_start, "end": slice_end, "reason": slice_reason},
                        "pre_sliced_clip_path": slice_clip_info.get("saved_clip_path", ""),
                    },
                    "trace": {"error": str(e)},
                }
            ],
        }
