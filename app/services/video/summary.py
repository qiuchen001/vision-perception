import json
import os
import re
from typing import Any

from openai import OpenAI

from app.algorithm.scene_mining.adapter import analyze_video, load_scene_mining_config, strip_model_thinking
from app.utils.logger import logger


SUMMARY_SYSTEM_PROMPT = (
    "你是自动驾驶场景视频分析总结助手。"
    "你会收到各类别视觉分析的结构化原始输出，请去除重复信息，"
    "按自然语言生成一段客观、简洁、可检索的视频内容总结。"
    "只输出JSON，不要输出思考过程。"
)

SUMMARY_MAX_CHARS = 800
SUMMARY_TARGET_CHARS = 300
SUMMARY_MAX_TOKENS = int(os.getenv("SCENE_MINING_SUMMARY_MAX_TOKENS", "4096"))
SUMMARY_RAW_OUTPUT_MAX_CHARS = int(os.getenv("SCENE_MINING_SUMMARY_RAW_OUTPUT_MAX_CHARS", "1500"))


SUMMARY_USER_PROMPT = (
    "请基于以下多类别分析结果生成最终内容总结。\n"
    "要求：\n"
    "1. 不要输出模型思考过程。\n"
    "2. 覆盖时间、光照、天气、道路环境、交通参与者、异常或风险事件。\n"
    "3. 若存在异常事件，说明大致时间段和风险类型。\n"
    f"4. 摘要必须简短，尽量不超过{SUMMARY_TARGET_CHARS}个中文字符，最多不超过{SUMMARY_MAX_CHARS}个中文字符。\n"
    "5. 禁止输出占位符、省略号或被截断的半句。\n"
    "6. 只输出 JSON 对象，包含 summary 字段，不要输出 Markdown 标题。\n\n"
    "{source_text}"
)

MIN_SUMMARY_CONTENT_CHARS = 12


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def _extract_summary_text(content: str) -> str:
    cleaned = _strip_code_fence(strip_model_thinking(content))
    parsed = None
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        json_match = re.search(r"\{[\s\S]*\}", cleaned)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                parsed = None
        else:
            parsed = None
        if parsed is None:
            if "Final Answer:" in cleaned:
                return cleaned.rsplit("Final Answer:", 1)[-1].strip()
            return cleaned
    if isinstance(parsed, dict):
        for key in ("summary", "summary_txt", "content", "text"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                return strip_model_thinking(value)
    return cleaned


def _limit_summary_text(text: str) -> str:
    summary = str(text or "").strip()
    if len(summary) <= SUMMARY_MAX_CHARS:
        return _ensure_sentence_end(summary)

    clipped = summary[:SUMMARY_MAX_CHARS]
    sentence_ends = [clipped.rfind(mark) for mark in ("。", "！", "？", ".", "!", "?")]
    end = max(sentence_ends)
    if end >= max(40, int(SUMMARY_MAX_CHARS * 0.6)):
        clipped = clipped[: end + 1]
    return _ensure_sentence_end(clipped.rstrip("，,；;、 "))


def _ensure_sentence_end(text: str) -> str:
    if not text:
        return text
    return text if text[-1] in "。！？.!?" else f"{text}。"


def _clip_source_text(text: str, max_chars: int) -> str:
    value = strip_model_thinking(str(text or "")).strip()
    if len(value) <= max_chars:
        return value
    clipped = value[:max_chars]
    sentence_ends = [clipped.rfind(mark) for mark in ("。", "！", "？", ".", "!", "?")]
    end = max(sentence_ends)
    if end >= max(80, int(max_chars * 0.5)):
        clipped = clipped[: end + 1]
    return _ensure_sentence_end(clipped.rstrip("，,；;、 "))


def _is_unusable_summary(text: str) -> bool:
    summary = str(text or "").strip()
    if not summary:
        return True
    if summary in {"...", "…", "……", "。。。"}:
        return True
    meaningful = re.sub(r"[\s.。!！?？,，;；、:：…-]+", "", summary)
    return len(meaningful) < MIN_SUMMARY_CONTENT_CHARS


def _pred_text(category_result: dict[str, Any]) -> str:
    pred = category_result.get("pred", [])
    if not isinstance(pred, list):
        return ""
    values = [str(item).strip() for item in pred if str(item).strip()]
    return "、".join(values)


def _build_rule_based_summary(scene_mining_result: dict[str, Any], category_order: list[str]) -> str:
    final_output = scene_mining_result.get("final_output", {})
    if not isinstance(final_output, dict):
        return ""

    env_categories = [
        "自然时间段",
        "人工光源辅助",
        "气象条件",
        "路面状态与视线干扰",
        "主干道路级别",
        "特殊路段与设施",
    ]
    env_parts: list[str] = []
    for category in env_categories:
        result = final_output.get(category)
        if isinstance(result, dict):
            text = _pred_text(result)
            if text:
                env_parts.append(text)

    sentences: list[str] = []
    if env_parts:
        sentences.append(f"视频场景为{'，'.join(env_parts)}")

    abnormal_events = scene_mining_result.get("abnormal_event_times") or []
    if not abnormal_events:
        summary_item = scene_mining_result.get("summary_item")
        if isinstance(summary_item, dict):
            abnormal_events = summary_item.get("abnormal_event_times") or []

    event_parts: list[str] = []
    if isinstance(abnormal_events, list):
        for event in abnormal_events:
            if not isinstance(event, dict):
                continue
            event_type = str(event.get("type") or "").strip()
            if not event_type:
                continue
            start = event.get("start_time")
            end = event.get("end_time")
            if isinstance(start, (int, float)) and isinstance(end, (int, float)):
                event_parts.append(f"{start:.1f}-{end:.1f}秒出现{event_type}")
            else:
                event_parts.append(event_type)

    if event_parts:
        sentences.append(f"异常或风险事件包括{'；'.join(event_parts)}")

    behavior_parts: list[str] = []
    for category in category_order:
        if category in env_categories or category == "静态障碍与全局异常":
            continue
        result = final_output.get(category)
        if not isinstance(result, dict):
            continue
        text = _pred_text(result)
        if text:
            behavior_parts.append(f"{category}为{text}")
    if behavior_parts:
        sentences.append("；".join(behavior_parts))

    return _limit_summary_text("。".join(sentences))


def _format_category_output(category: str, category_result: dict[str, Any], raw_output: str) -> str:
    parts = [f"【{category}】"]
    pred = category_result.get("pred", [])
    if isinstance(pred, list) and pred:
        parts.append(f"标签: {'、'.join(str(item) for item in pred)}")
    events = category_result.get("events", [])
    if isinstance(events, list) and events:
        parts.append(f"异常时间: {json.dumps(events, ensure_ascii=False)}")
    for key in (
        "step0_environment_context",
        "step0b_motion_reference",
        "step1_object_detection",
        "step2_motion_analysis",
        "step3_conflict_check",
    ):
        value = category_result.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(f"{key}: {strip_model_thinking(value)}")
    if raw_output:
        parts.append(f"raw_output: {raw_output}")
    return "\n".join(parts)


class SummaryVideoService:
    def summary(self, video_url: str, scene_mining_result: dict[str, Any] | None = None) -> dict[str, str]:
        if scene_mining_result is None:
            analyzed = analyze_video(video_url)
            scene_mining_result = {
                "final_output": analyzed.final_output,
                "raw_outputs": analyzed.raw_outputs,
            }

        config = load_scene_mining_config()
        final_output = scene_mining_result.get("final_output", {})
        raw_outputs = scene_mining_result.get("raw_outputs", {})
        if not isinstance(final_output, dict) or not final_output:
            raise ValueError("缺少场景挖掘结果，无法生成内容总结")
        if not isinstance(raw_outputs, dict):
            raw_outputs = {}

        category_order = config.get("pipeline", {}).get("category_order", list(final_output.keys()))
        missing_categories = [
            category
            for category in category_order
            if category not in final_output
        ]
        if missing_categories:
            raise ValueError(f"场景挖掘尚未完成，缺少类别: {', '.join(missing_categories)}")

        source_parts: list[str] = []
        for category in category_order:
            category_result = final_output.get(category)
            if not isinstance(category_result, dict):
                continue
            raw_output = _clip_source_text(raw_outputs.get(category, ""), SUMMARY_RAW_OUTPUT_MAX_CHARS)
            source_parts.append(_format_category_output(category, category_result, raw_output))
        source_text = "\n\n".join(source_parts)

        client = OpenAI(
            api_key=config["api"]["api_key"],
            base_url=config["api"]["base_url"],
        )
        response = client.chat.completions.create(
            model=config["api"]["model_name"],
            messages=[
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": SUMMARY_USER_PROMPT.format(source_text=source_text)},
            ],
            max_tokens=min(SUMMARY_MAX_TOKENS, int(config.get("model", {}).get("max_tokens", SUMMARY_MAX_TOKENS))),
            temperature=0.1,
            top_p=0.8,
            response_format={"type": "json_object"},
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        content = response.choices[0].message.content or ""
        summary_text = _extract_summary_text(content)
        if _is_unusable_summary(summary_text):
            logger.warning("Summary model returned unusable content: %r", summary_text)
            summary_text = _build_rule_based_summary(scene_mining_result, category_order)
        else:
            summary_text = _limit_summary_text(summary_text)
        if _is_unusable_summary(summary_text):
            raise ValueError("内容总结模型返回无效结果")
        if not summary_text:
            raise ValueError("内容总结模型返回空结果")
        return {"summary": summary_text}
