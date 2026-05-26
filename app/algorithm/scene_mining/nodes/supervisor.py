"""
Supervisor nodes for LangGraph pipeline.

supervisor_1: Generates simple_tasks list based on video metadata + previous_preds
supervisor_2: Generates complex_tasks time-slices using merged_simple results
"""

import asyncio
import json
import logging
import math
import re
from typing import Any

from prompts import load_skill
from qwen_client import QwenClient
from response_parser import ResponseParser

logger = logging.getLogger(__name__)


SUPERVISOR_SYSTEM_PROMPT = """你是一个任务规划专家，负责为视频分析生成任务列表。

根据输入的视频信息和已有结果，生成下一阶段的工作任务。

**规则：**
1. 只输出JSON格式的任务列表，不要输出其他内容
2. 每个任务必须包含 category 字段
3. 简单类别：自然时间段、人工光源辅助、气象条件、路面状态与视线干扰、主干道路级别、特殊路段与设施、静态障碍与全局异常
4. 复杂类别：弱势交通参与者异常、车辆轨迹与空间冲突、极端高危与失控事件
"""

# supervisor_2 LLM 模式专用 system prompt，从 skills/supervisor_2_system.md 加载
# 不继承 global_system（避免"只输出一个JSON对象"与数组输出冲突）
SUPERVISOR_2_SYSTEM_PROMPT = None  # lazy-loaded from skills
_SUPERVISOR_2_SYSTEM_PROMPT_FALLBACK = (
    "你是视频交通场景分析的时间窗规划专家。职责是目标检测，不是行为判断。"
    "调用 submit_time_plan 提交结果。skip/time_slices 规则详见工具参数说明。全程中文。"
)


def _get_supervisor_2_system_prompt(skills_dir: str = "skills") -> str:
    global SUPERVISOR_2_SYSTEM_PROMPT
    if SUPERVISOR_2_SYSTEM_PROMPT is not None:
        return SUPERVISOR_2_SYSTEM_PROMPT
    try:
        SUPERVISOR_2_SYSTEM_PROMPT = load_skill("supervisor_2_system", skills_dir)
    except FileNotFoundError:
        SUPERVISOR_2_SYSTEM_PROMPT = _SUPERVISOR_2_SYSTEM_PROMPT_FALLBACK
    return SUPERVISOR_2_SYSTEM_PROMPT

# 结构化工具 schema：强制 LLM 通过 function calling 输出，提升解析可靠性
# NOTE: 行为规则集中写在 tool description 中，system prompt 引用而不重复
SUPERVISOR_2_PLAN_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_time_plan",
        "description": (
            "提交该复杂类别的时间窗规划结果。\n"
            "规则：1) skip=true 仅当整段视频重点目标完全不存在且自车轨迹正常；"
            "2) 禁止因'目标行为未影响行车'使用 skip=true，行为判断是子Agent的工作；"
            "3) 若目标出现，必须 skip=false + time_slices。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "复杂类别名称"},
                "task_type": {"type": "string", "enum": ["complex"]},
                "need_subagent": {"type": "boolean"},
                "skip": {
                    "type": "boolean",
                    "description": "整段视频中明确无本类别重点目标时设为 true，此时 time_slices 可为空",
                },
                "skip_reason": {
                    "type": "string",
                    "description": "skip=true 时必填，说明为何确认无相关目标（须基于视觉证据）",
                },
                "relevant_context": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "从简单类别先验中摘取环境信息（天气/光照/道路类型），禁止填写候选子类名称或自行推断",
                },
                "time_slices": {
                    "type": "array",
                    "description": "最多 1 个时间窗口",
                    "items": {
                        "type": "object",
                        "properties": {
                            "start": {"type": "number"},
                            "end": {"type": "number"},
                            "suspected_evidence": {
                                "type": "string",
                                "description": (
                                    "仅限客观可见事实：目标种类 + 位置 + 朝向/运动方向。"
                                    "禁止任何行为结论、意图推断、交互描述（切入/横穿/逼近/冲突/阻断/威胁等词一律禁止）。"
                                    "若未见目标，应使用 skip=true，而非在此填写无目标说明。"
                                ),
                            },
                            "reason": {
                                "type": "string",
                                "description": (
                                    "说明为何选取该时间窗，仅限目标出现的时间段依据。"
                                    "禁止包含行为判断（切入/横穿/冲突/逼近等）。"
                                ),
                            },
                        },
                        "required": ["start", "end", "suspected_evidence", "reason"],
                    },
                },
            },
            "required": ["category", "task_type", "need_subagent", "relevant_context"],
        },
    },
}



SIMPLE_CATEGORY_TASKS = [
    {"category": "自然时间段", "task_type": "simple", "priority": 2, "notes": "基础环境先验"},
    {"category": "人工光源辅助", "task_type": "simple", "priority": 2, "notes": "光照补充信息"},
    {"category": "气象条件", "task_type": "simple", "priority": 2, "notes": "天气影响能见度"},
    {"category": "路面状态与视线干扰", "task_type": "simple", "priority": 3, "notes": "影响风险判断"},
    {"category": "主干道路级别", "task_type": "simple", "priority": 2, "notes": "道路场景类型"},
    {"category": "特殊路段与设施", "task_type": "simple", "priority": 3, "notes": "设施与临时区域"},
    {"category": "静态障碍与全局异常", "task_type": "simple", "priority": 3, "notes": "全局障碍首轮直出"},
]




def _extract_planner_focus_from_category_md(category_name: str, skills_dir: str) -> dict:
    """Extract focus_targets and ignore_targets from category .md file for Supervisor 2 planning.

    Reads '## Supervisor规划要点' as focus_targets (preferred), falling back to '## 分析要求'.
    Reads '## 忽略项' as ignore_targets.
    NOTE: uses the raw file content (before Supervisor节剥离) so supervisor sees its dedicated section.
    """
    from pathlib import Path as _Path
    import re as _re2

    path = _Path(skills_dir) / "categories" / f"{category_name}.md"
    try:
        content = path.read_text(encoding="utf-8").strip()
    except Exception:
        return {}
    if not content:
        return {}

    focus_targets = ""
    ignore_targets = ""

    # Prefer dedicated supervisor section; fall back to 分析要求
    supervisor_match = _re2.search(r'## Supervisor规划要点\s*\n(.*?)(?=\n## |\Z)', content, _re2.DOTALL)
    if supervisor_match:
        focus_targets = supervisor_match.group(1).strip()
    else:
        analysis_match = _re2.search(r'## 分析要求\s*\n(.*?)(?=\n## |\Z)', content, _re2.DOTALL)
        if analysis_match:
            focus_targets = analysis_match.group(1).strip()
        # Fallback: content between # title and first ## section
        if not focus_targets:
            fallback_match = _re2.search(r'# .+?\n\n(.*?)(?=\n## )', content, _re2.DOTALL)
            if fallback_match:
                focus_targets = fallback_match.group(1).strip()

    # Extract 忽略项 section
    ignore_match = _re2.search(r'## 忽略项\s*\n(.*?)(?=\n## |\Z)', content, _re2.DOTALL)
    if ignore_match:
        ignore_targets = ignore_match.group(1).strip()

    result = {}
    if focus_targets:
        result["focus_targets"] = focus_targets
    if ignore_targets:
        result["ignore_targets"] = ignore_targets
    return result


def _extract_content_or_reasoning(message) -> str:
    """Extract content from message, fallback to reasoning if content is None."""
    content = message.content
    if content:
        return content
    reasoning = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None)
    if reasoning:
        reasoning = re.sub(r"</?think>.*?</think>", "", reasoning, flags=re.DOTALL).strip()
        return reasoning
    return ""


def _load_supervisor_2_single_category_template(skills_dir: str) -> str:
    try:
        template = load_skill("supervisor_2_single_category_planner", skills_dir)
        if template:
            return template
    except Exception:
        pass
    logger.error("Failed to load supervisor_2_single_category_planner.md")
    return ""


def _normalize_cjk_quotes(text: str) -> str:
    """Replace CJK quotation marks with ASCII equivalents for JSON parsing.

    Delegates to ResponseParser to avoid duplicating the translation table.
    """
    return ResponseParser._normalize_cjk_quotes(text)


def _strip_thinking_preamble(content: str) -> str:
    """Strip plain-text thinking preamble that some reasoning models output.

    Handles both <think>...</think> tags and bare "Thinking Process:" style text.
    When thinking is detected, we keep only the content starting from the LAST
    '[{' pattern — that is most likely the actual JSON payload.
    """
    # Strip <think>...</think> block (greedy, in case of multiple)
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    # Detect common plain-text thinking markers
    _THINKING_MARKERS = re.compile(
        r"(?:Thinking Process\s*:|Let me (?:analyze|think)|让我(?:分析|思考|来)|Step\s*\d+\s*:|^\d+\.\s+\*\*)",
        re.IGNORECASE | re.MULTILINE,
    )
    if not _THINKING_MARKERS.search(content):
        return content

    # Find the LAST occurrence of '[{' or '[ {' — the real JSON array
    matches = list(re.finditer(r"\[\s*\{", content))
    if matches:
        return content[matches[-1].start():]

    return content


def _extract_json_array(content: str) -> list | None:
    """Extract JSON array from content that may contain thinking tags or preamble."""
    content = _strip_thinking_preamble(content)
    content_clean = content.strip()
    for start in range(len(content_clean)):
        if content_clean[start] == "[":
            depth = 0
            for i in range(start, len(content_clean)):
                if content_clean[i] == "[":
                    depth += 1
                elif content_clean[i] == "]":
                    depth -= 1
                if depth == 0:
                    try:
                        return json.loads(content_clean[start:i+1])
                    except json.JSONDecodeError:
                        continue
    return None


def _extract_tasks_from_tool_calls(message) -> list | None:
    """Extract task list from API tool_calls in the response message.

    Handles OpenAI-compatible tool_calls format where the model returns
    structured data via function calling instead of plain text.
    """
    tool_calls = getattr(message, "tool_calls", None)
    if not tool_calls:
        return None
    for tc in tool_calls:
        func = getattr(tc, "function", None)
        if not func:
            continue
        args_str = getattr(func, "arguments", "")
        if not args_str:
            continue
        for text in (args_str, _normalize_cjk_quotes(args_str)):
            try:
                args = json.loads(text)
                if isinstance(args, list):
                    return args
                if isinstance(args, dict):
                    for key in ("tasks", "complex_tasks", "data"):
                        value = args.get(key)
                        if isinstance(value, list):
                            return value
                    if "category" in args or "time_slices" in args or "skip" in args:
                        return [args]
            except (json.JSONDecodeError, ValueError):
                continue
    return None


def _strip_content_wrappers(content: str) -> str:
    """Remove common LLM output wrappers: markdown code blocks, <answer> tags, <tool> tags."""
    content = re.sub(r"```(?:json)?\s*", "", content)
    content = re.sub(r"```\s*", "", content)
    content = re.sub(r"</?answer>", "", content, flags=re.IGNORECASE)
    content = re.sub(r"</?tool>", "", content)
    return content.strip()


def _extract_tasks_flexible(content: str) -> list:
    """Multi-strategy extraction of task lists from LLM output.

    Strategies (in order):
    1. Direct JSON array parse (raw + CJK-normalized)
    2. Strip wrappers and retry
    3. Balanced-brace dict extraction with known-key unwrapping
    """
    if not content:
        return []

    # --- Phase 1: Direct JSON array extraction (raw + CJK-normalized) ---
    for text in (content, _normalize_cjk_quotes(content)):
        tasks = _extract_json_array(text)
        if isinstance(tasks, list):
            return tasks

    # --- Phase 2: Strip wrappers and retry ---
    stripped = _strip_content_wrappers(content)
    if stripped != content:
        for text in (stripped, _normalize_cjk_quotes(stripped)):
            tasks = _extract_json_array(text)
            if isinstance(tasks, list):
                return tasks

    # --- Phase 3: Balanced-brace dict extraction with known-key unwrapping ---
    content_clean = _strip_thinking_preamble(content or "").strip()
    for start in range(len(content_clean)):
        if content_clean[start] != "{":
            continue
        depth = 0
        for i in range(start, len(content_clean)):
            if content_clean[i] == "{":
                depth += 1
            elif content_clean[i] == "}":
                depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(content_clean[start:i + 1])
                except json.JSONDecodeError:
                    break
                if isinstance(obj, dict):
                    for key in ("tasks", "complex_tasks", "data"):
                        value = obj.get(key)
                        if isinstance(value, list):
                            return value
                break
    return []


_MIN_SLICE_DURATION = 2.0  # 时间切片最小有效时长（秒）


def _validate_task_format(task: dict, duration: float = 0.0, window_seconds: float = 0.0) -> tuple[bool, str]:
    """校验任务格式，返回 (是否有效, 错误信息)。

    触发 repair 的错误（模型需重新提交）：
    - 必需字段缺失
    - task_type 不为 "complex"
    - time_slices 不是合法对象数组
    - start >= end
    - start >= 视频时长（时间戳超出视频范围）
    - 窗口时长 < _MIN_SLICE_DURATION（过短，无法有效分析）
    - relevant_context 不是数组

    不触发 repair（由归一化阶段处理）：
    - 窗口过长（> window_seconds）→ 归一化阶段 clamp
    - 全局窗口（覆盖 ≥80% 视频）→ 归一化阶段拆分
    """
    if not isinstance(task, dict):
        return False, f"任务应为 dict，实际为 {type(task).__name__}"

    required_fields = ["category", "task_type", "need_subagent"]
    for field in required_fields:
        if field not in task:
            return False, f"缺少必需字段: {field}"

    task_type = task.get("task_type", "")
    if task_type != "complex":
        return False, f"task_type 应为 'complex'，实际为 '{task_type}'"

    # skip=True 时 time_slices 可为空/缺失
    if task.get("skip") is True:
        return True, ""

    # time_slices 必须是对象数组（非 skip 时）
    time_slices = task.get("time_slices", [])
    if not isinstance(time_slices, list):
        return False, f"time_slices 应为数组，实际为 {type(time_slices).__name__}"

    if not time_slices:
        return False, "time_slices 不能为空（如确认无目标请使用 skip=true）"

    for i, sl in enumerate(time_slices):
        if not isinstance(sl, dict):
            return False, f"time_slices[{i}] 应为对象，实际为 {type(sl).__name__}"
        if "start" not in sl or "end" not in sl:
            return False, f"time_slices[{i}] 缺少 start 或 end 字段"
        try:
            start_val = float(sl["start"])
            end_val = float(sl["end"])
        except (TypeError, ValueError) as e:
            return False, f"time_slices[{i}] start/end 非数字: {e}"

        if start_val >= end_val:
            return False, f"time_slices[{i}] start({start_val:.2f}) >= end({end_val:.2f})"

        # 时间戳超出视频范围：模型出现了幻觉时间戳，必须 repair
        if duration > 0 and start_val >= duration:
            return False, (
                f"time_slices[{i}] start({start_val:.1f}s) 超出视频时长({duration:.1f}s)，"
                f"请在 0 ~ {duration:.1f}s 范围内重新选取时间窗"
            )

        # 窗口过短：即使能归一化展开，中心点仍可能是模型幻觉，必须 repair
        window_len = end_val - start_val
        if window_len < _MIN_SLICE_DURATION:
            return False, (
                f"time_slices[{i}] 窗口时长({window_len:.2f}s)过短，"
                f"最小需 {_MIN_SLICE_DURATION:.0f}s，请给出包含目标出现时段的有效时间窗"
            )

    # 窗口过长：警告即可，归一化阶段会 clamp，不触发 repair
    if window_seconds > 0:
        for i, sl in enumerate(time_slices):
            try:
                window_len = float(sl["end"]) - float(sl["start"])
                if window_len > window_seconds:
                    logger.debug(
                        "Supervisor 2: time_slices[%d] 长度 %.1fs 超过 %.1fs，将在归一化阶段 clamp",
                        i, window_len, window_seconds,
                    )
            except (TypeError, ValueError):
                pass

    # relevant_context 必须是数组（如果存在）
    if "relevant_context" in task and task["relevant_context"] is not None:
        rc = task["relevant_context"]
        if not isinstance(rc, list):
            return False, f"relevant_context 应为数组，实际为 {type(rc).__name__}"

    return True, ""


def _centered_window(duration: float, center: float, window: float) -> tuple[float, float]:
    total = max(0.1, float(duration or 0.0))
    w = max(1.0, min(float(window or 8.0), total))
    c = max(0.0, min(float(center), total))
    left = max(0.0, c - w / 2.0)
    right = min(total, left + w)
    left = max(0.0, right - w)
    return left, max(left + 0.1, right)




def _parse_time_slices_string(raw: str) -> list[dict]:
    """Try to parse malformed time_slices strings like '8.0-14.0' or '00:01:23-00:01:25'."""
    import re
    results: list[dict] = []
    # Split by common separators: comma or semicolon for multiple slices
    parts = re.split(r"[,;]", raw)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Try "start-end" format with seconds
        m = re.match(r"(\d+\.?\d*)\s*[-~]\s*(\d+\.?\d*)", part)
        if m:
            try:
                results.append({"start": float(m.group(1)), "end": float(m.group(2))})
                continue
            except (ValueError, TypeError):
                pass
        # Try "HH:MM:SS-HH:MM:SS" format
        m2 = re.match(
            r"(\d{1,2}):(\d{2}):(\d{2}(?:\.\d+)?)\s*[-~]\s*(\d{1,2}):(\d{2}):(\d{2}(?:\.\d+)?)",
            part,
        )
        if m2:
            try:
                s = int(m2.group(1)) * 3600 + int(m2.group(2)) * 60 + float(m2.group(3))
                e = int(m2.group(4)) * 3600 + int(m2.group(5)) * 60 + float(m2.group(6))
                results.append({"start": s, "end": e})
                continue
            except (ValueError, TypeError):
                pass
    return results


def _build_uniform_time_slices(
    duration: float,
    window_seconds: float,
    max_slices_per_task: int,
    step_seconds: float = 0.0,
    start_offset: float = 0.0,
) -> list[dict]:
    total = max(0.1, float(duration or 0.0))
    window = max(2.0, float(window_seconds or 8.0))
    if window > total:
        window = total
    # step=0 means non-overlapping (step == window); otherwise sliding window
    step = float(step_seconds) if step_seconds and step_seconds > 0 else window
    step = max(1.0, min(step, window))

    if total <= window:
        total_windows = 1
    else:
        total_windows = math.ceil((total - window) / step) + 1

    all_slices: list[dict] = []
    for idx in range(total_windows):
        start = start_offset + idx * step
        end = min(start_offset + total, start + window)
        if end <= start:
            continue
        all_slices.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "reason": f"全局扫描窗口 {idx + 1}/{total_windows}",
            }
        )
    limit = max(1, int(max_slices_per_task or 1))
    if len(all_slices) <= limit:
        return all_slices
    if limit == 1:
        mid = len(all_slices) // 2
        return [all_slices[mid]]
    picked_indices = sorted(
        {
            int(round(i * (len(all_slices) - 1) / (limit - 1)))
            for i in range(limit)
        }
    )
    if len(picked_indices) < limit:
        for idx in range(len(all_slices)):
            if idx not in picked_indices:
                picked_indices.append(idx)
            if len(picked_indices) >= limit:
                break
        picked_indices = sorted(set(picked_indices))
    return [all_slices[idx] for idx in picked_indices[:limit]]


def _window_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    if union <= 0:
        return 0.0
    return inter / union


def _window_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _merge_slices_by_iou(
    slices: list[dict],
    duration: float,
    window_seconds: float,
    iou_threshold: float,
) -> list[dict]:
    if not isinstance(slices, list) or len(slices) < 2:
        return slices
    # Ignore tiny float-boundary overlaps (e.g. 8.0 vs 7.999989).
    # Otherwise adjacent windows can be merged unexpectedly.
    min_overlap_for_merge = 0.05
    # Kept iou_threshold argument for backward compatibility with callers/config.
    merged_items: list[dict] = []
    for item in slices:
        if not isinstance(item, dict):
            continue
        try:
            start_val = float(item.get("start"))
            end_val = float(item.get("end"))
        except (TypeError, ValueError):
            continue
        if end_val <= start_val:
            continue
        merged_items.append(
            {
                "start": start_val,
                "end": end_val,
                "reason": str(item.get("reason", "") or ""),
            }
        )
    if len(merged_items) < 2:
        return merged_items
    duration_bound = float(duration or 0.0)
    if duration_bound <= 0:
        duration_bound = max(max(float(x["end"]) for x in merged_items), float(window_seconds or 8.0))

    changed = True
    guard = 0
    while changed and len(merged_items) > 1 and guard < 20:
        guard += 1
        changed = False
        merged_items.sort(key=lambda x: (float(x["start"]), float(x["end"])))
        pair_found = None
        for i in range(len(merged_items)):
            for j in range(i + 1, len(merged_items)):
                left = merged_items[i]
                right = merged_items[j]
                overlap = _window_overlap(
                    float(left["start"]),
                    float(left["end"]),
                    float(right["start"]),
                    float(right["end"]),
                )
                if overlap >= min_overlap_for_merge:
                    pair_found = (i, j, overlap)
                    break
            if pair_found is not None:
                break
        if pair_found is None:
            break
        i, j, overlap = pair_found
        a = merged_items[i]
        b = merged_items[j]
        center = (
            ((float(a["start"]) + float(a["end"])) / 2.0)
            + ((float(b["start"]) + float(b["end"])) / 2.0)
        ) / 2.0
        new_start, new_end = _centered_window(
            duration=duration_bound,
            center=center,
            window=window_seconds,
        )
        reason_a = str(a.get("reason", "") or "").strip()
        reason_b = str(b.get("reason", "") or "").strip()
        reasons = [x for x in [reason_a, reason_b] if x]
        merged_reason = "；".join(reasons) if reasons else ""
        if merged_reason:
            merged_reason = f"{merged_reason}；时间交叉({overlap:.2f}s)合并"
        else:
            merged_reason = f"时间交叉({overlap:.2f}s)合并"
        new_item = {"start": new_start, "end": new_end, "reason": merged_reason}
        remained = [x for idx, x in enumerate(merged_items) if idx not in (i, j)]
        remained.append(new_item)
        merged_items = remained
        changed = True
    merged_items.sort(key=lambda x: (float(x["start"]), float(x["end"])))
    return merged_items


def _build_code_complex_tasks(
    merged_simple: dict,
    duration: float,
    window_seconds: float,
    max_slices_per_task: int = 1,
    step_seconds: float = 0.0,
) -> list[dict]:
    def _preds(category: str) -> list[str]:
        if not isinstance(merged_simple, dict):
            return []
        value = merged_simple.get(category, {})
        if isinstance(value, dict):
            pred = value.get("pred", [])
            if isinstance(pred, list):
                return [str(item).strip() for item in pred if str(item).strip()]
        return []

    total = max(0.1, float(duration or 0.0))
    window = max(2.0, float(window_seconds or 8.0))
    if window > total:
        window = total

    time_preds = set(_preds("自然时间段"))
    weather_preds = set(_preds("气象条件"))
    road_preds = set(_preds("主干道路级别"))
    facility_preds = set(_preds("特殊路段与设施"))

    is_closed_road = bool({"城市快速路/高架路", "高速公路"} & road_preds)
    is_closed_space = bool({"隧道内"} & facility_preds)
    is_low_light = bool({"夜晚", "晨昏"} & time_preds)
    is_bad_weather = bool({"雨天", "雪天", "雾霾天", "沙尘天气"} & weather_preds)

    base_context: list[str] = []
    if is_closed_road:
        base_context.append("封闭路段")
    if is_closed_space:
        base_context.append("隧道内")
    if is_low_light:
        base_context.append("低照度")
    if is_bad_weather:
        base_context.append("不良天气")

    slices = _build_uniform_time_slices(
        duration=total,
        window_seconds=window,
        max_slices_per_task=max_slices_per_task,
        step_seconds=step_seconds,
    )

    tasks: list[dict] = []
    for category in ["弱势交通参与者异常", "车辆轨迹与空间冲突", "极端高危与失控事件"]:
        if is_closed_road or is_closed_space:
            suspected = "封闭路段/空间，重点关注逆行、倒车、异常停车、掉落物等高危行为"
            priority = 1
        elif is_low_light or is_bad_weather:
            suspected = "低照度/不良天气，关注目标持续轨迹而非单帧特征"
            priority = 2
        else:
            suspected = "常规场景，全局扫描后按窗口复核"
            priority = 2
        task_slices = [{**item, "suspected_evidence": suspected} for item in slices]
        tasks.append(
            {
                "category": category,
                "task_type": "complex",
                "need_subagent": True,
                "time_slices": task_slices,
                "priority": priority,
                "relevant_context": list(base_context),
            }
        )
    return tasks


async def supervisor_1_node(state: dict) -> dict:
    """
    Supervisor 1: Generate simple_tasks list.

    Simple categories are fixed — no VLM call needed.

    Output state keys set:
    - simple_tasks: list[dict]
    """
    simple_tasks = [dict(item) for item in SIMPLE_CATEGORY_TASKS]
    logger.info("Supervisor 1 generated %d simple tasks", len(simple_tasks))
    return {
        "simple_tasks": simple_tasks,
        "agent_outputs": [
            {
                "agent_name": "supervisor_1_simple_task_planner",
                "task_name": "simple_task_planning",
                "raw_content": "generated_by_code",
                "result": simple_tasks,
                "trace": {"mode": "code", "reason": "fixed_simple_categories"},
            }
        ],
    }


async def supervisor_2_node(state: dict) -> dict:
    """
    Supervisor 2: Generate complex_tasks list based on merged simple results.

    Input state keys used:
    - video_path: str
    - video_info: dict
    - merged_simple: dict

    Output state keys set:
    - complex_tasks: list[dict]
    """
    video_path = state["video_path"]
    on_step = state.get("on_step")
    if on_step:
        on_step("abnormal_planning", "异常事件分析中... 规划时间窗", {})
    video_url = state["video_url"]
    video_info = state.get("video_info", {})
    duration = video_info.get("duration", 0)
    config = state.get("config", {})
    skills_dir = state.get("skills_dir", "skills")
    complex_supervisor_mode = str(
        config.get("pipeline", {}).get("complex_supervisor_mode", "code")
    ).lower()
    first_round_cfg = (
        config.get("tool_call", {})
        .get("fixed_sampling", {})
        .get("profiles", {})
        .get("complex", {})
        .get("first_round", {})
    )
    default_window = float(first_round_cfg.get("window_seconds", 8.0) or 8.0)
    default_step = float(first_round_cfg.get("window_step_seconds", 0.0) or 0.0)
    if default_step <= 0:
        default_step = default_window * 0.75  # default: 75% of window → 6s for 8s window
    merge_iou_threshold = float(first_round_cfg.get("slice_merge_iou_threshold", 0.4) or 0.4)
    max_slices_per_task = max(
        1,
        int(config.get("pipeline", {}).get("complex_max_slices_per_task", 3) or 3),
    )
    merged_simple = state.get("merged_simple", {})

    client: QwenClient = state["qwen_client"]

    # ---- YOLO mode: use YOLO pre-filter results ----
    yolo_detections = state.get("yolo_detections", {})
    yolo_cfg = config.get("yolo", {})
    yolo_enabled = bool(yolo_cfg.get("enabled", False))

    if complex_supervisor_mode == "yolo" and yolo_enabled and yolo_detections:
        complex_direct_results: dict[str, dict] = {}
        complex_tasks: list[dict] = []

        # Build base context from merged_simple (same logic as _build_code_complex_tasks)
        def _preds(category: str) -> list[str]:
            if not isinstance(merged_simple, dict):
                return []
            value = merged_simple.get(category, {})
            if isinstance(value, dict):
                pred = value.get("pred", [])
                if isinstance(pred, list):
                    return [str(item).strip() for item in pred if str(item).strip()]
            return []

        time_preds = set(_preds("自然时间段"))
        weather_preds = set(_preds("气象条件"))
        road_preds = set(_preds("主干道路级别"))
        facility_preds = set(_preds("特殊路段与设施"))

        is_closed_road = bool({"城市快速路/高架路", "高速公路"} & road_preds)
        is_closed_space = bool({"隧道内"} & facility_preds)
        is_low_light = bool({"夜晚", "晨昏"} & time_preds)
        is_bad_weather = bool({"雨天", "雪天", "雾霾天", "沙尘天气"} & weather_preds)

        base_context: list[str] = []
        if is_closed_road:
            base_context.append("封闭路段")
        if is_closed_space:
            base_context.append("隧道内")
        if is_low_light:
            base_context.append("低照度")
        if is_bad_weather:
            base_context.append("不良天气")

        for category in ["弱势交通参与者异常", "车辆轨迹与空间冲突", "极端高危与失控事件"]:
            yolo_windows = yolo_detections.get(category, [])

            if not yolo_windows:
                # YOLO detected nothing suspicious → directly output normal result, skip LLM
                default_pred = ResponseParser.get_default_normal_pred(category)
                complex_direct_results[category] = {
                    "pred": default_pred if isinstance(default_pred, list) else [default_pred],
                    "events": [],
                    "step1_object_detection": "",
                    "step2_motion_analysis": "",
                    "step3_conflict_check": "",
                }
                logger.info(
                    "Supervisor 2 [YOLO] [%s]: no detections → direct normal pred",
                    category,
                )
                continue

            # Enrich YOLO windows with context
            if is_closed_road or is_closed_space:
                suspected = "封闭路段/空间，重点关注逆行、倒车、异常停车、掉落物等高危行为"
            elif is_low_light or is_bad_weather:
                suspected = "低照度/不良天气，关注目标持续轨迹而非单帧特征"
            else:
                suspected = "YOLO预筛选可疑时间窗，需LLM精细确认"

            enriched_slices: list[dict] = []
            for w in yolo_windows[:max_slices_per_task]:
                enriched_slices.append({
                    "start": float(w.get("start", 0)),
                    "end": float(w.get("end", 0)),
                    "reason": str(w.get("reason", "YOLO pre-filter window")),
                    "suspected_evidence": str(w.get("suspected_evidence", suspected)),
                })

            # Apply IoU merge and window clamping
            enriched_slices = _merge_slices_by_iou(
                slices=enriched_slices,
                duration=float(duration or 0.0),
                window_seconds=default_window,
                iou_threshold=merge_iou_threshold,
            )
            clamped_slices: list[dict] = []
            for sl in enriched_slices[:max_slices_per_task]:
                try:
                    center = (float(sl["start"]) + float(sl["end"])) / 2.0
                    left, right = _centered_window(
                        duration=float(duration or 0.0),
                        center=center,
                        window=default_window,
                    )
                    clamped_slices.append({
                        "start": left,
                        "end": max(left + 0.1, right),
                        "reason": str(sl.get("reason", "")),
                        "suspected_evidence": str(sl.get("suspected_evidence", suspected)),
                    })
                except (TypeError, ValueError):
                    clamped_slices.append(sl)

            complex_tasks.append({
                "category": category,
                "task_type": "complex",
                "need_subagent": True,
                "time_slices": clamped_slices,
                "priority": 1,
                "relevant_context": list(base_context),
            })

        logger.info(
            "Supervisor 2 [YOLO mode]: %d tasks + %d direct normal results",
            len(complex_tasks), len(complex_direct_results),
        )
        return {
            "complex_tasks": complex_tasks,
            "complex_direct_results": complex_direct_results,
            "agent_outputs": [
                {
                    "agent_name": "supervisor_2_complex_task_planner",
                    "task_name": "complex_task_planning",
                    "raw_content": "yolo_pre_filter",
                    "result": {
                        "complex_tasks": complex_tasks,
                        "complex_direct_results": complex_direct_results,
                    },
                    "trace": {
                        "mode": "yolo",
                        "reason": "yolo_pre_filter_windows",
                        "yolo_detections_summary": {
                            cat: len(wins) for cat, wins in yolo_detections.items()
                        },
                    },
                }
            ],
        }

    # ---- Fallback: YOLO mode but no detections or YOLO disabled → code mode ----
    if complex_supervisor_mode == "yolo" and (not yolo_enabled or not yolo_detections):
        logger.info(
            "Supervisor 2 [YOLO mode]: no YOLO results, falling back to code mode"
        )

    if complex_supervisor_mode != "llm":
        complex_tasks = _build_code_complex_tasks(
            merged_simple=merged_simple,
            duration=float(duration or 0.0),
            window_seconds=default_window,
            max_slices_per_task=max_slices_per_task,
            step_seconds=default_step,
        )
        logger.info("Supervisor 2 generated %d complex tasks by code", len(complex_tasks))
        return {
            "complex_tasks": complex_tasks,
            "complex_direct_results": {},
            "agent_outputs": [
                {
                    "agent_name": "supervisor_2_complex_task_planner",
                    "task_name": "complex_task_planning",
                    "raw_content": "generated_by_code",
                    "result": {"complex_tasks": complex_tasks, "complex_direct_results": {}},
                    "trace": {"mode": "code", "reason": "programmatic_uniform_window_planning"},
                }
            ],
        }

    # Build merged simple results context (exclude "静态障碍与全局异常" to avoid prior bias)
    _EXCLUDED_FROM_PRIOR = {"静态障碍与全局异常"}
    if merged_simple:
        context_parts = ["简单类别分析结果："]
        for cat, result in merged_simple.items():
            if cat in _EXCLUDED_FROM_PRIOR:
                continue
            pred = result.get("pred", []) if isinstance(result, dict) else result
            context_parts.append(f"- {cat}: {json.dumps(pred, ensure_ascii=False)}")
        merged_context = "\n".join(context_parts)
    else:
        merged_context = "无简单类别结果"

    video_cfg = config.get("video", {})
    complex_category_list = ["弱势交通参与者异常", "车辆轨迹与空间冲突", "极端高危与失控事件"]
    complex_categories_set = set(complex_category_list)
    default_video_fps = video_info.get("fps", 1)
    planner_focus = {}
    for cat in complex_category_list:
        focus_cfg = _extract_planner_focus_from_category_md(cat, skills_dir)
        if focus_cfg:
            planner_focus[cat] = focus_cfg
    single_category_template = _load_supervisor_2_single_category_template(skills_dir)

    try:
        planner_concurrency = max(
            1,
            int(config.get("pipeline", {}).get("complex_planner_concurrency", len(complex_category_list)) or len(complex_category_list)),
        )
        planner_semaphore = asyncio.Semaphore(planner_concurrency)

        def _pick_task(extracted: list, category_name: str) -> dict | None:
            """Pick the best-matching task dict from extracted list."""
            for item in extracted:
                if isinstance(item, dict) and item.get("category") == category_name:
                    return dict(item)
            for item in extracted:
                if isinstance(item, dict) and item.get("category") in (None, ""):
                    return dict(item)
            if extracted and isinstance(extracted[0], dict):
                return dict(extracted[0])
            return None

        def _extract_from_message(message) -> list:
            """Try tool_calls first (primary), then text JSON (fallback)."""
            tasks = _extract_tasks_from_tool_calls(message)
            if tasks:
                return tasks
            content = _extract_content_or_reasoning(message)
            return _extract_tasks_flexible(content)

        async def _call_api(msgs: list, max_tok: int, temp: float) -> object:
            """Call API; use native tool calling only when enabled and supported."""
            use_native = bool(
                client.config.get("react_agent", {}).get("use_native_tools", True)
            )
            if use_native:
                try:
                    return await client.client.chat.completions.create(
                        model=client.config["api"]["model_name"],
                        messages=msgs,
                        max_tokens=max_tok,
                        temperature=temp,
                        tools=[SUPERVISOR_2_PLAN_TOOL],
                        # "auto" instead of forced function: model can emit text + tool_call,
                        # preventing content=None which crashes vLLM's _parse_tool_calls_from_content
                        tool_choice="auto",
                    )
                except Exception as exc:
                    logger.info("Supervisor 2: tool calling unsupported (%s), falling back to plain", exc)
            return await client.client.chat.completions.create(
                model=client.config["api"]["model_name"],
                messages=msgs,
                max_tokens=max_tok,
                temperature=temp,
            )

        async def _plan_single_category(category_name: str) -> tuple[str, str, dict]:
            async with planner_semaphore:
                content = ""
                max_full_retries = int(
                    config.get("retry", {}).get("max_supervisor2_retries", 3)
                )
                base_temperature = float(client.config.get("model", {}).get("temperature", 0.1))
                focus_cfg = planner_focus.get(category_name, {})
                user_prompt = single_category_template.format(
                    category=category_name,
                    video_path=video_path,
                    duration=duration,
                    merged_simple_results=merged_context,
                    focus_targets=focus_cfg.get("focus_targets", "- 对应类别核心目标"),
                    ignore_targets=focus_cfg.get("ignore_targets", "- 非该类别目标"),
                )
                messages = [
                    {"role": "system", "content": _get_supervisor_2_system_prompt(skills_dir)},
                    {
                        "role": "user",
                        "content": [
                            {"type": "video_url", "video_url": {"url": video_url}},
                            {"type": "text", "text": user_prompt},
                        ],
                    },
                ]

                for api_attempt in range(max_full_retries):
                    try:
                        # Keep temperature constant: format compliance degrades with higher temperature
                        temperature = base_temperature
                        if api_attempt > 0:
                            logger.info(
                                "Supervisor 2 [%s]: full API retry %d/%d (temperature=%.2f)",
                                category_name, api_attempt + 1, max_full_retries, temperature,
                            )

                        response = await _call_api(messages, 8192, temperature)
                        message = response.choices[0].message
                        content = _extract_content_or_reasoning(message)
                        extracted = _extract_from_message(message)

                        # --- Format validation ---
                        format_error = ""
                        if extracted:
                            candidate = _pick_task(extracted, category_name)
                            if candidate is not None:
                                is_valid, error_msg = _validate_task_format(
                                    candidate, duration=float(duration or 0.0), window_seconds=default_window
                                )
                                if not is_valid:
                                    format_error = error_msg
                                    logger.info("Supervisor 2 [%s]: format invalid: %s", category_name, error_msg)
                                    extracted = []

                        # --- Repair rounds (continue original conversation WITH video, tool calling still active) ---
                        max_repair_rounds = 2
                        for repair_round in range(max_repair_rounds):
                            if extracted:
                                candidate = _pick_task(extracted, category_name)
                                if candidate is not None:
                                    ok, err = _validate_task_format(
                                        candidate, duration=float(duration or 0.0), window_seconds=default_window
                                    )
                                    if ok:
                                        break
                                    format_error = err
                                    extracted = []
                                else:
                                    break

                            repair_prompt = (
                                f"你刚才对类别「{category_name}」的输出存在格式问题，请基于你刚才观察到的视频内容，调用工具 submit_time_plan 重新提交。\n"
                            )
                            if format_error:
                                repair_prompt += f"【格式错误】{format_error}\n"
                            repair_prompt += (
                                f'category="{category_name}", task_type="complex"。其余字段要求详见工具参数说明。\n'
                            )
                            # Build repair as a continuation of the original conversation so the model
                            # retains full video context. Include the failed assistant turn so it can
                            # see what it got wrong rather than generating from scratch.
                            tool_calls = getattr(message, "tool_calls", None)
                            if tool_calls:
                                asst_msg: dict = {
                                    "role": "assistant",
                                    "content": content or None,
                                    "tool_calls": [
                                        {
                                            "id": tc.id,
                                            "type": "function",
                                            "function": {
                                                "name": tc.function.name,
                                                "arguments": tc.function.arguments,
                                            },
                                        }
                                        for tc in tool_calls
                                    ],
                                }
                                tool_result_msgs = [
                                    {
                                        "role": "tool",
                                        "tool_call_id": tc.id,
                                        "content": f"格式校验失败: {format_error}",
                                    }
                                    for tc in tool_calls
                                ]
                                repair_messages = messages + [asst_msg] + tool_result_msgs + [
                                    {"role": "user", "content": repair_prompt}
                                ]
                            else:
                                asst_msg = {"role": "assistant", "content": content or ""}
                                repair_messages = messages + [asst_msg, {"role": "user", "content": repair_prompt}]
                            repair_response = await _call_api(repair_messages, 4096, temperature)
                            repair_message = repair_response.choices[0].message
                            extracted = _extract_from_message(repair_message)
                            if extracted:
                                # Update content so planner_raw_by_category reflects the actual output
                                repair_content = _extract_content_or_reasoning(repair_message)
                                if repair_content:
                                    content = repair_content
                                logger.info(
                                    "Supervisor 2 [%s]: repair round %d succeeded", category_name, repair_round + 1
                                )
                            else:
                                format_error = "修复后仍无法解析"

                        # Final pick
                        picked_task = _pick_task(extracted, category_name) if extracted else None

                        if picked_task is not None:
                            picked_task["category"] = category_name
                            # When native tool calling is used, content is often empty
                            # (all output goes into tool_calls). Serialize picked_task
                            # so planner_raw_by_category has meaningful data for debugging.
                            if not content:
                                content = json.dumps(picked_task, ensure_ascii=False)
                            return category_name, content, picked_task

                        # This attempt (including its repair rounds) produced nothing usable.
                        logger.warning(
                            "Supervisor 2 [%s]: parse attempt %d/%d failed (raw preview: %s)%s",
                            category_name,
                            api_attempt + 1,
                            max_full_retries,
                            (content or "")[:200],
                            ", retrying full API call..." if api_attempt + 1 < max_full_retries else ", no more retries.",
                        )

                    except Exception as exc:
                        logger.warning(
                            "Supervisor 2 [%s] attempt %d/%d exception: %s",
                            category_name, api_attempt + 1, max_full_retries, exc,
                        )

                # All full retries exhausted — fall back to default "no anomaly" result.
                logger.warning(
                    "Supervisor 2 [%s]: all %d API retries exhausted, using fallback",
                    category_name, max_full_retries,
                )
                return category_name, content, None

        complex_direct_results: dict[str, dict] = {}
        planned_results = await asyncio.gather(
            *(_plan_single_category(category_name) for category_name in complex_category_list)
        )
        planner_raw_by_category: dict[str, str] = {}
        planner_tasks: list[dict] = []
        for category_name, content, picked_task in planned_results:
            planner_raw_by_category[category_name] = content
            if picked_task is None:
                # 规划解析失败，直接输出默认"无异常"结果，不走 sub-agent
                default_pred = ResponseParser.get_default_normal_pred(category_name)
                complex_direct_results[category_name] = {
                    "pred": default_pred if isinstance(default_pred, list) else [default_pred],
                    "events": [],
                    "step1_object_detection": "",
                    "step2_motion_analysis": "",
                    "step3_conflict_check": "",
                }
                logger.info(
                    "Supervisor 2 [%s]: planner failed, using default pred %s (no sub-agent)",
                    category_name,
                    default_pred,
                )
            else:
                planner_tasks.append(picked_task)

        tasks = planner_tasks
        complex_tasks: list[dict] = []
        for t in tasks:
            if not isinstance(t, dict):
                continue
            category = t.get("category")
            if category not in complex_categories_set:
                continue

            # --- Skip: supervisor confirmed no relevant target ---
            if t.get("skip") is True:
                skip_reason = str(t.get("skip_reason", "") or "无相关目标")
                logger.info("Supervisor 2 [%s]: skip=true, reason=%s", category, skip_reason)
                default_pred = ResponseParser.get_default_normal_pred(category)
                complex_direct_results[category] = {
                    "pred": default_pred if isinstance(default_pred, list) else [default_pred],
                    "events": [],
                    "step1_object_detection": "",
                    "step2_motion_analysis": "",
                    "step3_conflict_check": skip_reason,
                }
                continue

            need_subagent = t.get("need_subagent")
            if isinstance(need_subagent, bool):
                need_subagent_bool = need_subagent
            else:
                need_subagent_bool = True
            suspected_text = str(t.get("suspected_evidence", "") or "")
            if not suspected_text:
                raw_slices_for_evidence = t.get("time_slices") or []
                if isinstance(raw_slices_for_evidence, list) and raw_slices_for_evidence and isinstance(raw_slices_for_evidence[0], dict):
                    suspected_text = str(raw_slices_for_evidence[0].get("suspected_evidence", "") or "")
            # 强制 need_subagent=True：禁止全局复核，必须通过子agent复核
            if not need_subagent_bool:
                need_subagent_bool = True
            raw_slices = t.get("time_slices")
            # Handle malformed time_slices: string like "00:01:23-00:01:25" or "8.0-14.0"
            if isinstance(raw_slices, str):
                raw_slices = _parse_time_slices_string(raw_slices)
            if not isinstance(raw_slices, list) or not raw_slices:
                raw_single = t.get("time_slice")
                if isinstance(raw_single, dict):
                    raw_slices = [raw_single]
                elif isinstance(raw_single, str):
                    raw_slices = _parse_time_slices_string(raw_single)
                else:
                    raw_slices = []
            normalized_slices = []
            for item in raw_slices:
                if not isinstance(item, dict):
                    continue
                try:
                    start_val = float(item.get("start"))
                    end_val = float(item.get("end"))
                except (TypeError, ValueError):
                    continue
                if end_val <= start_val:
                    continue
                item_reason = str(item.get("reason", "") or "")
                item_evidence = str(item.get("suspected_evidence", "") or "")

                dur_f = float(duration) if duration and duration > 0 else 0.0
                # 大窗口（覆盖 ≥80% 视频）：用滑动窗口展开，size=window, step<window 保证相邻覆盖
                is_global = dur_f > 0 and (end_val - start_val) / dur_f >= 0.80
                if is_global:
                    uniform = _build_uniform_time_slices(
                        duration=dur_f,
                        window_seconds=default_window,
                        max_slices_per_task=max_slices_per_task,
                        step_seconds=default_step,
                    )
                    for u in uniform:
                        normalized_slices.append({
                            "start": u["start"],
                            "end": u["end"],
                            "reason": item_reason or u.get("reason", "全局扫描拆分"),
                            "suspected_evidence": item_evidence or "未见明确可疑目标，做全局均匀扫描",
                        })
                    continue

                # 模型自行决定时间范围，前后各扩展 1s 作为缓冲，再做边界 clamp
                _PAD = 1.0
                start_val = start_val - _PAD
                end_val = end_val + _PAD
                if dur_f > 0:
                    start_val = max(0.0, min(start_val, dur_f))
                    end_val = max(start_val + 0.1, min(end_val, dur_f))
                else:
                    start_val = max(0.0, start_val)
                    end_val = max(start_val + 0.1, end_val)
                normalized_slices.append(
                    {
                        "start": start_val,
                        "end": end_val,
                        "reason": item_reason,
                        "suspected_evidence": item_evidence,
                    }
                )
            if not normalized_slices:
                # 无有效切片，直接输出默认"无异常"结果，不走 sub-agent
                logger.info("Supervisor 2 [%s]: 无有效切片，直接输出默认结果", category)
                default_pred = ResponseParser.get_default_normal_pred(category)
                complex_direct_results[category] = {
                    "pred": default_pred if isinstance(default_pred, list) else [default_pred],
                    "events": [],
                    "step1_object_detection": "",
                    "step2_motion_analysis": "",
                    "step3_conflict_check": "",
                }
                continue
            # Note: do NOT merge normalized slices here — the LLM planner already chose
            # non-overlapping windows intentionally. Merging after _centered_window expansion
            # collapses adjacent 8s windows (e.g. [0-8],[5-13],[8-16] → 1 slice).
            normalized_task = dict(t)
            normalized_task["need_subagent"] = True
            normalized_task["time_slices"] = normalized_slices[:max_slices_per_task]
            complex_tasks.append(normalized_task)

        logger.info(
            "Supervisor 2 generated %d complex tasks (all require subagent review)",
            len(complex_tasks),
        )
        return {
            "complex_tasks": complex_tasks,
            "complex_direct_results": complex_direct_results,
            "agent_outputs": [
                {
                    "agent_name": "supervisor_2_complex_task_planner",
                    "task_name": "complex_task_planning",
                    "raw_content": json.dumps(planner_raw_by_category, ensure_ascii=False),
                    "result": {"complex_tasks": complex_tasks, "complex_direct_results": complex_direct_results},
                    "trace": {"all_extracted_tasks": tasks, "planner_raw_by_category": planner_raw_by_category},
                }
            ],
        }

    except Exception as e:
        logger.error("Supervisor 2 error: %s", e)
        return {
            "complex_tasks": [],
            "complex_direct_results": {},
            "agent_outputs": [
                {
                    "agent_name": "supervisor_2_complex_task_planner",
                    "task_name": "complex_task_planning",
                    "raw_content": "",
                    "result": [],
                    "trace": {"error": str(e)},
                }
            ],
        }
