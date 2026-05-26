"""
Output Formatter Node for LangGraph pipeline.

Combines all predictions into final result.json format.
Saves per-category raw output and trace files.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _temporal_iou(a: dict, b: dict) -> float:
    """Compute temporal IoU between two events."""
    a_s, a_e = float(a.get("start_time", 0.0)), float(a.get("end_time", 0.0))
    b_s, b_e = float(b.get("start_time", 0.0)), float(b.get("end_time", 0.0))
    overlap = max(0.0, min(a_e, b_e) - max(a_s, b_s))
    span_a, span_b = a_e - a_s, b_e - b_s
    union = span_a + span_b - overlap
    if union <= 0:
        return 0.0
    return overlap / union


def _temporal_overlap(a: dict, b: dict) -> float:
    a_s, a_e = float(a.get("start_time", 0.0)), float(a.get("end_time", 0.0))
    b_s, b_e = float(b.get("start_time", 0.0)), float(b.get("end_time", 0.0))
    return max(0.0, min(a_e, b_e) - max(a_s, b_s))


def _nms_events_within_category(events: list[dict], iou_threshold: float = 0.5) -> list[dict]:
    """NMS for events within a single category.

    Same-type overlapping events are merged (union of time windows).
    Different-type events are kept even if they overlap (they describe different things).
    """
    if not events:
        return events

    # Group by type
    type_groups: dict[str, list[dict]] = {}
    for evt in events:
        evt_type = str(evt.get("type", ""))
        type_groups.setdefault(evt_type, []).append(evt)

    result: list[dict] = []
    for evt_type, group in type_groups.items():
        if len(group) <= 1:
            result.extend(group)
            continue

        # Sort by start_time
        group.sort(key=lambda e: float(e.get("start_time", 0.0)))

        # Merge overlapping events of the same type
        merged = [dict(group[0])]
        for evt in group[1:]:
            last = merged[-1]
            overlap = _temporal_overlap(last, evt)
            if overlap > 0:
                # Merge: union of time windows
                last["start_time"] = min(float(last["start_time"]), float(evt.get("start_time", 0.0)))
                last["end_time"] = max(float(last["end_time"]), float(evt.get("end_time", 0.0)))
            else:
                merged.append(dict(evt))

        result.extend(merged)

    # Sort by start_time
    result.sort(key=lambda e: float(e.get("start_time", 0.0)))
    return result


def _mark_cross_category_relations(final_output: dict, iou_threshold: float = 0.5) -> None:
    """Detect cross-category event correlations based on temporal overlap.

    Adds a 'related' field to each event listing related event types from other categories.
    """
    # Collect all events with their category
    all_events: list[tuple[str, int, dict]] = []
    for cat, cat_result in final_output.items():
        if not isinstance(cat_result, dict):
            continue
        events = cat_result.get("events", [])
        if not isinstance(events, list):
            continue
        for idx, evt in enumerate(events):
            if not isinstance(evt, dict):
                continue
            all_events.append((cat, idx, evt))

    # Find cross-category overlaps
    relations: dict[tuple[str, int], list[str]] = {}
    for i, (cat_i, idx_i, evt_i) in enumerate(all_events):
        for j, (cat_j, idx_j, evt_j) in enumerate(all_events):
            if j <= i:
                continue
            if cat_i == cat_j:
                continue
            if _temporal_iou(evt_i, evt_j) > iou_threshold:
                key_i = (cat_i, idx_i)
                key_j = (cat_j, idx_j)
                evt_j_type = str(evt_j.get("type", ""))
                evt_i_type = str(evt_i.get("type", ""))
                relations.setdefault(key_i, []).append(f"{cat_j}/{evt_j_type}")
                relations.setdefault(key_j, []).append(f"{cat_i}/{evt_i_type}")

    # Write related field back into events
    for cat, idx, evt in all_events:
        related = relations.get((cat, idx))
        if related:
            # evt is a reference into final_output, mutation works
            evt["related"] = related


def _normalize_event_time_range(start_val: float, end_val: float, duration: float) -> tuple[float, float]:
    start = float(start_val)
    end = float(end_val)
    if end < start:
        start, end = end, start
    if duration <= 0:
        return max(0.0, start), max(0.0, end)
    span = max(0.1, end - start)
    if end <= duration and start >= 0:
        return start, end
    if start >= duration:
        new_end = duration
        new_start = max(0.0, duration - span)
        return new_start, max(new_start + 0.1, new_end)
    new_start = max(0.0, start)
    new_end = min(duration, end)
    if new_end <= new_start:
        new_end = min(duration, new_start + 0.1)
    return new_start, new_end


def _normalize_events(events: list, duration: float) -> list[dict]:
    normalized: list[dict] = []
    for item in events if isinstance(events, list) else []:
        if not isinstance(item, dict):
            continue
        evt = dict(item)
        try:
            start_val = float(evt.get("start_time", 0.0))
            end_val = float(evt.get("end_time", start_val))
        except (TypeError, ValueError):
            continue
        start_val, end_val = _normalize_event_time_range(start_val, end_val, duration)
        evt["start_time"] = start_val
        evt["end_time"] = end_val
        normalized.append(evt)
    return normalized


def _extract_complex_preds_from_agent_outputs(agent_outputs: list) -> list[dict]:
    """Rebuild complex worker outputs from agent_outputs records.

    This provides a second source-of-truth when aggregated `complex_preds`
    count is inconsistent with expected slices.
    """
    rebuilt: list[dict] = []
    if not isinstance(agent_outputs, list):
        return rebuilt
    for item in agent_outputs:
        if not isinstance(item, dict):
            continue
        agent_name = str(item.get("agent_name", "") or "")
        if not agent_name.startswith("complex_worker_"):
            continue
        task_name = str(item.get("task_name", "") or "")
        if " [" in task_name:
            category = task_name.split(" [", 1)[0].strip()
        else:
            # Fallback: complex_worker_{category}_{sub_id}
            tail = agent_name[len("complex_worker_") :]
            category = tail.rsplit("_", 1)[0] if "_" in tail else tail
        result = item.get("result", {})
        if not isinstance(result, dict):
            continue
        rebuilt.append(
            {
                "category": category,
                "pred": result.get("pred", []) if isinstance(result.get("pred", []), list) else [],
                "events": result.get("events", []) if isinstance(result.get("events", []), list) else [],
                "time_slice": result.get("time_slice", {}) if isinstance(result.get("time_slice", {}), dict) else {},
                # Agent outputs currently don't persist step texts per slice.
                "step1_object_detection": "",
                "step2_motion_analysis": "",
                "step3_conflict_check": "",
            }
        )
    return rebuilt


def _choose_complex_preds_source(
    expected_complex_slices: int,
    complex_preds_from_state: list,
    complex_preds_from_agent_outputs: list,
) -> tuple[list, str]:
    """Choose the most complete complex preds source."""
    state_count = len(complex_preds_from_state) if isinstance(complex_preds_from_state, list) else 0
    agent_count = len(complex_preds_from_agent_outputs) if isinstance(complex_preds_from_agent_outputs, list) else 0
    expected = max(0, int(expected_complex_slices or 0))

    # No strict expectation -> prefer non-empty state payload, then fallback.
    if expected <= 0:
        if state_count > 0:
            return complex_preds_from_state, "state(complex_preds)"
        if agent_count > 0:
            return complex_preds_from_agent_outputs, "agent_outputs(rebuilt)"
        return [], "empty"

    # Prefer whichever source fully matches expected slices.
    state_ok = state_count == expected
    agent_ok = agent_count == expected
    if state_ok and not agent_ok:
        return complex_preds_from_state, "state(complex_preds)"
    if agent_ok and not state_ok:
        return complex_preds_from_agent_outputs, "agent_outputs(rebuilt)"
    if state_ok and agent_ok:
        return complex_preds_from_state, "state(complex_preds)"

    # If both incomplete, pick the larger one and log upstream.
    if agent_count > state_count:
        return complex_preds_from_agent_outputs, "agent_outputs(rebuilt,incomplete)"
    return complex_preds_from_state, "state(complex_preds,incomplete)"


async def output_formatter_node(state: dict) -> dict:
    """
    Combine all predictions into final output.
    Saves raw.txt and trace.json for each category.

    Input state keys used:
    - merged_simple: dict (simple category results)
    - complex_preds: list[dict] (complex category results)
    - simple_traces: list[dict] (simple category traces)
    - complex_traces: list[dict] (complex category traces)
    - output_dir: str

    Output state keys set:
    - final_output: dict (full result in existing result.json format)
    """
    merged_simple = state.get("merged_simple", {})
    complex_preds = state.get("complex_preds", [])
    complex_direct_results = state.get("complex_direct_results", {})
    simple_traces = state.get("simple_traces", [])
    complex_traces = state.get("complex_traces", [])
    agent_outputs = state.get("agent_outputs", [])
    expected_complex_slices = int(state.get("expected_complex_slices", 0) or 0)
    output_dir = state.get("output_dir", "")
    video_info = state.get("video_info", {})
    try:
        video_duration = float(video_info.get("duration", 0.0)) if isinstance(video_info, dict) else 0.0
    except (TypeError, ValueError):
        video_duration = 0.0

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Build category -> trace mapping
    trace_map: dict[str, dict] = {}
    for t in simple_traces:
        cat = t.get("category")
        if cat:
            if cat in trace_map:
                logger.info("Trace map overwrite for category (simple): %s", cat)
            trace_map[cat] = {"raw_content": t.get("raw_content", ""), "trace": t.get("trace", {})}
    for t in complex_traces:
        cat = t.get("category")
        if cat:
            if cat in trace_map:
                logger.info("Trace map overwrite for category (complex): %s", cat)
            trace_map[cat] = {"raw_content": t.get("raw_content", ""), "trace": t.get("trace", {})}

    # Start with simple category results
    final_output = {}

    for category, result in merged_simple.items():
        if isinstance(result, dict):
            normalized_result = dict(result)
            normalized_result["events"] = _normalize_events(
                normalized_result.get("events", []),
                video_duration,
            )
            final_output[category] = normalized_result
        else:
            final_output[category] = {"pred": result}

        # Save raw.txt and trace.json for simple category
        if category in trace_map:
            trace_data = trace_map[category]
            raw_file = output_path / f"{category}_raw.txt"
            raw_file.write_text(trace_data.get("raw_content", "") or "", encoding="utf-8")
            trace_file = output_path / f"{category}_trace.json"
            trace_file.write_text(
                json.dumps(trace_data.get("trace", {}), ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            logger.debug("Saved trace files for simple category: %s", category)

    if isinstance(complex_direct_results, dict):
        for category, result in complex_direct_results.items():
            if not isinstance(result, dict):
                continue
            final_output[category] = {
                "pred": result.get("pred", []),
                "events": _normalize_events(result.get("events", []), video_duration),
                "step1_object_detection": result.get("step1_object_detection", ""),
                "step2_motion_analysis": result.get("step2_motion_analysis", ""),
                "step3_conflict_check": result.get("step3_conflict_check", ""),
            }

    # Add complex category results (merge multi sub-agents by category)
    rebuilt_complex_preds = _extract_complex_preds_from_agent_outputs(agent_outputs)
    selected_complex_preds, complex_source = _choose_complex_preds_source(
        expected_complex_slices=expected_complex_slices,
        complex_preds_from_state=complex_preds if isinstance(complex_preds, list) else [],
        complex_preds_from_agent_outputs=rebuilt_complex_preds,
    )
    if expected_complex_slices > 0:
        logger.info(
            "Complex source selection: expected=%d, state_preds=%d, rebuilt_from_agent_outputs=%d, using=%s",
            expected_complex_slices,
            len(complex_preds) if isinstance(complex_preds, list) else 0,
            len(rebuilt_complex_preds),
            complex_source,
        )

    complex_grouped: dict[str, list[dict]] = {}
    for pred_item in selected_complex_preds:
        if not isinstance(pred_item, dict):
            continue
        category = pred_item.get("category")
        if not category:
            continue
        complex_grouped.setdefault(category, []).append(pred_item)

        # Save raw.txt and trace.json for complex category
        if category in trace_map:
            trace_data = trace_map[category]
            raw_file = output_path / f"{category}_raw.txt"
            raw_file.write_text(trace_data.get("raw_content", "") or "", encoding="utf-8")
            trace_file = output_path / f"{category}_trace.json"
            trace_file.write_text(
                json.dumps(trace_data.get("trace", {}), ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            logger.debug("Saved trace files for complex category: %s", category)

    for category, items in complex_grouped.items():
        merged_pred: list[str] = []
        merged_events: list[dict] = []
        step1_parts: list[str] = []
        step2_parts: list[str] = []
        step3_parts: list[str] = []
        abnormal_preds = {"无异常交互", "无异常车辆轨迹冲突", "无极端高危事件", "道路情况正常",
                          "无明显人工光源", "无特殊情况", "无异常"}
        abnormal_flags: list[bool] = []
        for item in items:
            item_preds: list[str] = item.get("pred", []) if isinstance(item, dict) else []
            is_abnormal_item = any(p not in abnormal_preds for p in item_preds)
            abnormal_flags.append(is_abnormal_item)

        has_abnormal = any(abnormal_flags)
        for idx, item in enumerate(items):
            if has_abnormal and not abnormal_flags[idx]:
                continue
            time_slice = item.get("time_slice", {}) if isinstance(item, dict) else {}
            start_val = float(time_slice.get("start", 0.0)) if isinstance(time_slice, dict) else 0.0
            end_val = float(time_slice.get("end", 0.0)) if isinstance(time_slice, dict) else 0.0
            tag = f"[{start_val:.2f}s-{end_val:.2f}s]"
            item_preds: list[str] = item.get("pred", []) if isinstance(item, dict) else []
            for p in item_preds:
                if p not in merged_pred:
                    merged_pred.append(p)
            merged_events.extend(item.get("events", []) if isinstance(item, dict) else [])
            s1 = str(item.get("step1_object_detection", "") or "").strip() if isinstance(item, dict) else ""
            s2 = str(item.get("step2_motion_analysis", "") or "").strip() if isinstance(item, dict) else ""
            s3 = str(item.get("step3_conflict_check", "") or "").strip() if isinstance(item, dict) else ""
            if s1:
                step1_parts.append(f"{tag} {s1}")
            if s2:
                step2_parts.append(f"{tag} {s2}")
            if s3:
                step3_parts.append(f"{tag} {s3}")

        normalized_events = _normalize_events(merged_events, video_duration)
        deduped_events = _nms_events_within_category(normalized_events)
        final_output[category] = {
            "pred": merged_pred,
            "events": deduped_events,
            "step1_object_detection": "\n".join(step1_parts),
            "step2_motion_analysis": "\n".join(step2_parts),
            "step3_conflict_check": "\n".join(step3_parts),
        }

    complex_agent_outputs_count = 0
    if isinstance(agent_outputs, list):
        for item in agent_outputs:
            if not isinstance(item, dict):
                continue
            agent_name = str(item.get("agent_name", ""))
            if agent_name.startswith("complex_worker_"):
                complex_agent_outputs_count += 1

    if expected_complex_slices > 0:
        logger.info(
            "Complex aggregation check: expected_slices=%d, selected_complex_preds=%d, complex_agent_outputs=%d",
            expected_complex_slices,
            len(selected_complex_preds),
            complex_agent_outputs_count,
        )

    agent_output_map = {}
    for item in agent_outputs:
        if not isinstance(item, dict):
            continue
        agent_name = item.get("agent_name")
        if not agent_name:
            continue
        # Avoid silent overwrite when duplicate agent_name appears.
        if agent_name in agent_output_map:
            dup_idx = 2
            new_name = f"{agent_name}__dup{dup_idx}"
            while new_name in agent_output_map:
                dup_idx += 1
                new_name = f"{agent_name}__dup{dup_idx}"
            logger.warning("Duplicate agent_name detected in agent_outputs: %s -> %s", agent_name, new_name)
            agent_name = new_name
        agent_output_map[agent_name] = item
        safe_name = (
            str(agent_name)
            .replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
            .replace(" ", "_")
        )
        raw_text = item.get("raw_content", "") or ""
        trace_obj = item.get("trace", {})
        (output_path / f"{safe_name}_raw.txt").write_text(raw_text, encoding="utf-8")
        (output_path / f"{safe_name}_trace.json").write_text(
            json.dumps(trace_obj, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    (output_path / "agent_outputs.json").write_text(
        json.dumps(agent_output_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Cross-category event correlation
    _mark_cross_category_relations(final_output)

    # Fill missing categories with default "normal" result
    all_expected_categories = [
        "自然时间段", "人工光源辅助", "气象条件", "路面状态与视线干扰",
        "主干道路级别", "特殊路段与设施", "静态障碍与全局异常",
        "弱势交通参与者异常", "车辆轨迹与空间冲突", "极端高危与失控事件",
    ]
    default_normal_preds = {
        "自然时间段": ["白天"],
        "人工光源辅助": ["无明显人工光源"],
        "气象条件": ["晴天"],
        "路面状态与视线干扰": ["路面干燥正常"],
        "主干道路级别": ["城市地面道路"],
        "特殊路段与设施": ["无特殊情况"],
        "静态障碍与全局异常": ["道路情况正常"],
        "弱势交通参与者异常": ["无异常交互"],
        "车辆轨迹与空间冲突": ["无异常车辆轨迹冲突"],
        "极端高危与失控事件": ["无极端高危事件"],
    }
    for cat in all_expected_categories:
        if cat not in final_output:
            default_pred = default_normal_preds.get(cat, [])
            result = {"pred": default_pred, "events": []}
            if cat in ("弱势交通参与者异常", "车辆轨迹与空间冲突", "极端高危与失控事件", "静态障碍与全局异常"):
                result["step1_object_detection"] = ""
                result["step2_motion_analysis"] = ""
                result["step3_conflict_check"] = ""
            final_output[cat] = result
            logger.info("Filled missing category: %s with default normal", cat)

    logger.info("Output formatter: produced final output with %d categories", len(final_output))
    return {"final_output": final_output}
