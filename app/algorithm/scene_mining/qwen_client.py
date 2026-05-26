import asyncio
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import unquote, urlparse

import httpx
from openai import AsyncOpenAI, BadRequestError

from response_parser import ResponseParser

logger = logging.getLogger(__name__)


CLIP_SELECT_TOOL = {
    "type": "function",
    "function": {
        "name": "clip_select",
        "description": "截取视频的局部时间窗口进行精细分析。证据充分时直接输出JSON，无需强制调用。",
        "parameters": {
            "type": "object",
            "properties": {
                "start_time": {"type": "number", "description": "截取起始时间（秒）"},
                "end_time": {"type": "number", "description": "截取结束时间（秒），单窗口不超过8秒"},
            },
            "required": ["start_time", "end_time"],
        },
    },
}


class QwenClient:

    # Loaded lazily from skills/ on first access via _get_complex_prompt()
    _COMPLEX_FIRST_ROUND_PROMPT_FALLBACK = (
        "complex类别首轮规则：不要输出最终JSON，"
        "先调用 clip_select 工具定位可疑时间窗。"
        "拿到切片后若有把握再输出JSON，否则继续追加局部切片。"
    )
    _COMPLEX_GLOBAL_REJECTED_PROMPT_FALLBACK = (
        "已获得全局视频信息，complex类别首次clip_select不能做全局切片。"
        "请只针对可疑事件窗口进行局部切片。"
    )

    def _get_complex_prompt(self, prompt_key: str) -> str:
        """Load a complex-worker prompt from skills/, falling back to hardcoded default."""
        from prompts import load_skill
        skills_dir = self.config.get("paths", {}).get("skills", "skills")
        try:
            return load_skill(prompt_key, skills_dir)
        except FileNotFoundError:
            if prompt_key == "complex_first_round":
                return self._COMPLEX_FIRST_ROUND_PROMPT_FALLBACK
            elif prompt_key == "complex_global_rejected":
                return self._COMPLEX_GLOBAL_REJECTED_PROMPT_FALLBACK
            return ""

    def __init__(self, config: dict):
        self.config = config
        api_config = config["api"]
        timeout = api_config["timeout"]
        max_connections = int(api_config.get("httpx_max_connections", 1000))
        max_keepalive_connections = int(
            api_config.get("httpx_max_keepalive_connections", 1000)
        )
        keepalive_expiry = float(api_config.get("httpx_keepalive_expiry", 120.0))
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_keepalive_connections,
                keepalive_expiry=keepalive_expiry,
            ),
        )
        self.client = AsyncOpenAI(
            api_key=api_config["api_key"],
            base_url=api_config["base_url"],
            timeout=timeout,
            http_client=self.http_client,
        )
        self._system_prompt: str = ""
        self._simple_system_prompt: str = ""
        self._complex_system_prompt: str = ""

    def init_system_prompt(self, prompt: str) -> None:
        self._system_prompt = prompt

    def init_split_system_prompts(self, simple_prompt: str, complex_prompt: str) -> None:
        self._simple_system_prompt = simple_prompt
        self._complex_system_prompt = complex_prompt

    async def analyze_category_sequential(
        self,
        video_url: str,
        category_name: str,
        category_prompt: str,
        previous_preds: dict,
        output_dir: str,
        is_abnormal: bool = False,
        detailed: bool = False,
        pre_sliced_input: bool = False,
        sampling_round_offset: int = 0,
        global_time_offset: float = 0.0,
        video_duration: float = 0.0,
    ) -> tuple[str, str, dict, dict]:
        return await self._analyze_with_parse_retry(
            video_url=video_url,
            category_prompt=category_prompt,
            category_name=category_name,
            previous_preds=previous_preds,
            output_dir=output_dir,
            is_abnormal=is_abnormal,
            detailed=detailed,
            pre_sliced_input=pre_sliced_input,
            sampling_round_offset=sampling_round_offset,
            global_time_offset=global_time_offset,
            video_duration=video_duration,
        )

    async def _analyze_with_parse_retry(
        self,
        video_url: str,
        category_prompt: str,
        category_name: str,
        previous_preds: dict,
        output_dir: str,
        is_abnormal: bool,
        detailed: bool,
        pre_sliced_input: bool,
        sampling_round_offset: int,
        global_time_offset: float,
        video_duration: float = 0.0,
    ) -> tuple[str, str, dict, dict]:
        max_parse_retries = self.config["retry"]["max_parse_retries"]
        last_raw = ""
        last_trace: dict = {"rounds": []}
        result: dict = {
            "category": category_name,
            "error": "no_attempts",
            "pred": [],
            "_parse_ok": False,
        }
        if is_abnormal:
            result["events"] = []

        for attempt in range(max_parse_retries + 1):
            if attempt > 0:
                logger.warning(
                    "JSON解析失败 [%s] 第%d次尝试，重新发送原始视频和提示词",
                    category_name,
                    attempt,
                )

            raw_content, trace = await self._call_api_with_retry(
                video_url=video_url,
                category_prompt=category_prompt,
                category_name=category_name,
                previous_preds=previous_preds,
                output_dir=output_dir,
                parse_attempt=attempt,
                is_abnormal=is_abnormal,
                detailed=detailed,
                pre_sliced_input=pre_sliced_input,
                sampling_round_offset=sampling_round_offset,
                global_time_offset=global_time_offset,
                video_duration=video_duration,
            )
            last_raw = raw_content
            last_trace = trace

            if raw_content.startswith("ERROR:"):
                self._append_parse_debug_log(
                    output_dir=output_dir,
                    video_url=video_url,
                    category_name=category_name,
                    parse_attempt=attempt,
                    stage="api_or_repair_error",
                    raw_content=raw_content,
                    parse_result={
                        "category": category_name,
                        "error": "api_error",
                        "_retry_count": attempt,
                    },
                )
                # Unrecoverable errors (e.g. video stream inaccessible): bail immediately
                if trace.get("_unrecoverable") or attempt >= max_parse_retries:
                    default_pred = ResponseParser.get_default_normal_pred(category_name)
                    fallback_result = {
                        "category": category_name,
                        "error": "api_error",
                        "pred": default_pred,
                        "_retry_count": attempt,
                        "_fallback_applied": True,
                    }
                    if is_abnormal:
                        fallback_result["events"] = []
                    return category_name, raw_content, fallback_result, last_trace
                continue

            result = ResponseParser.parse(category_name, raw_content, attempt, is_abnormal=is_abnormal)

            if not ResponseParser.is_parse_failed(result):
                return category_name, raw_content, result, last_trace
            self._append_parse_debug_log(
                output_dir=output_dir,
                video_url=video_url,
                category_name=category_name,
                parse_attempt=attempt,
                stage="parse_failed",
                raw_content=raw_content,
                parse_result=result,
            )

            if attempt < max_parse_retries:
                await asyncio.sleep(1)
            else:
                logger.error(
                    "JSON解析失败 [%s] 已达最大重试次数，使用兜底结果", category_name,
                )

                try:
                    failed_log_path = Path(output_dir).parent / "failed_videos.log"
                    with failed_log_path.open("a", encoding="utf-8") as f:
                        f.write(f"[{datetime.now().isoformat()}] 解析彻底失败: {video_url} | 类别: {category_name}\n")
                        f.write(f"最后一次尝试的原始输出:\n{last_raw}\n")
                        f.write("-" * 80 + "\n")
                except Exception as e:
                    logger.warning("写入failed_videos.log失败: %s", e)

                default_pred = ResponseParser.get_default_normal_pred(category_name)
                result["pred"] = default_pred
                if is_abnormal:
                    result["events"] = []
                result["_fallback_applied"] = True

                self._append_parse_debug_log(
                    output_dir=output_dir,
                    video_url=video_url,
                    category_name=category_name,
                    parse_attempt=attempt,
                    stage="fallback_returned",
                    raw_content=last_raw,
                    parse_result=result,
                )
                return category_name, last_raw, result, last_trace

        return category_name, last_raw, result, last_trace

    def _append_parse_debug_log(
        self,
        output_dir: str,
        video_url: str,
        category_name: str,
        parse_attempt: int,
        stage: str,
        raw_content: str,
        parse_result: dict | None = None,
    ) -> None:
        try:
            output_path = Path(output_dir)
            run_output_dir = output_path.parent
            log_file = run_output_dir / "parse_retry_debug.jsonl"
            log_item = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "video_url": video_url,
                "category": category_name,
                "parse_attempt": parse_attempt,
                "stage": stage,
                "raw_content": raw_content,
                "parse_result": parse_result,
            }
            with log_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(log_item, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning("写入解析调试日志失败: %s", exc)

    async def _call_api_with_retry(
        self,
        video_url: str,
        category_prompt: str,
        category_name: str,
        previous_preds: dict,
        output_dir: str,
        parse_attempt: int = 0,
        is_abnormal: bool = False,
        detailed: bool = False,
        pre_sliced_input: bool = False,
        sampling_round_offset: int = 0,
        global_time_offset: float = 0.0,
        video_duration: float = 0.0,
    ) -> tuple[str, dict]:
        max_api_retries = self.config["retry"]["max_api_retries"]
        last_error: Exception | None = None

        for attempt in range(max_api_retries + 1):
            try:
                raw, trace = await self._run_tool_call_session(
                    video_url=video_url,
                    category_prompt=category_prompt,
                    category_name=category_name,
                    previous_preds=previous_preds,
                    output_dir=output_dir,
                    parse_attempt=parse_attempt,
                    is_abnormal=is_abnormal,
                    detailed=detailed,
                    pre_sliced_input=pre_sliced_input,
                    sampling_round_offset=sampling_round_offset,
                    global_time_offset=global_time_offset,
                    video_duration=video_duration,
                )
                return raw, trace
            except Exception as exc:
                last_error = exc
                exc_type = type(exc).__name__
                logger.warning(
                    "API调用失败 [%s] 第%d次尝试: [%s] %s", category_name, attempt + 1, exc_type, exc
                )
                # 视频流无法访问/解码属于不可恢复错误，立即终止重试
                if isinstance(exc, BadRequestError) and "video" in str(exc).lower():
                    logger.error("视频流不可恢复错误，跳过重试: %s", exc)
                    return (
                        f"ERROR: API调用失败，已达最大重试次数 [{exc_type}] {exc}",
                        {"rounds": [], "_unrecoverable": True},
                    )
                if attempt < max_api_retries:
                    await asyncio.sleep(2 ** attempt)

        exc_type = type(last_error).__name__ if last_error else "Unknown"
        exc_msg = str(last_error) if last_error else ""
        return f"ERROR: API调用失败，已达最大重试次数 [{exc_type}] {exc_msg}", {"rounds": []}

    async def _run_tool_call_session(
        self,
        video_url: str,
        category_prompt: str,
        category_name: str,
        previous_preds: dict,
        output_dir: str,
        parse_attempt: int = 0,
        is_abnormal: bool = False,
        detailed: bool = False,
        pre_sliced_input: bool = False,
        sampling_round_offset: int = 0,
        global_time_offset: float = 0.0,
        video_duration: float = 0.0,
    ) -> tuple[str, dict]:
        messages = self._build_category_messages(
            video_url, category_prompt, category_name, previous_preds,
            is_abnormal=is_abnormal,
        )
        max_turns = int(self.config.get("tool_call", {}).get("max_turns", 6))
        trace: dict = {"rounds": []}
        last_content = ""
        tool_used = False
        format_retry_injected = False
        first_tool_reminder_sent = False  # guard: _COMPLEX_FIRST_ROUND_PROMPT only once
        current_global_offset = float(global_time_offset)
        # Use the caller-supplied duration; fallback to 0 (will be probed lazily on first tool call).
        current_video_duration = float(video_duration) if video_duration > 0 else 0.0
        category_complexity = self._resolve_category_complexity(category_name)
        # Native tool calling enabled by default; auto-disabled on API error or config
        tool_call_enabled = self.config.get("tool_call", {}).get("complex_worker_tool_call_enabled", True)
        use_native_tools = tool_call_enabled and not pre_sliced_input
        require_first_tool = category_complexity == "complex" and use_native_tools
        if parse_attempt > 0:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": ResponseParser.generate_retry_prompt(is_abnormal=is_abnormal),
                        }
                    ],
                }
            )
            format_retry_injected = True
        if require_first_tool:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self._get_complex_prompt("complex_first_round"),
                        }
                    ],
                }
            )
            first_tool_reminder_sent = True

        for round_idx in range(max_turns):
            effective_round_idx = max(0, round_idx + sampling_round_offset)
            tools_param = [CLIP_SELECT_TOOL] if (use_native_tools and not pre_sliced_input) else None
            try:
                response = await self._async_call_api(
                    messages=messages,
                    category_name=category_name,
                    parse_attempt=parse_attempt,
                    round_idx=effective_round_idx,
                    duration=current_video_duration,
                    tools=tools_param,
                )
            except Exception as exc:
                err_str = str(exc).lower()
                if use_native_tools and any(k in err_str for k in ("tool choice", "tool_choice", "tool-call-parser", "function call")):
                    logger.warning("[%s] Native tool calling rejected by server, falling back to XML mode", category_name)
                    use_native_tools = False
                    response = await self._async_call_api(
                        messages=messages,
                        category_name=category_name,
                        parse_attempt=parse_attempt,
                        round_idx=effective_round_idx,
                        duration=current_video_duration,
                    )
                else:
                    raise

            message = response.choices[0].message
            content = message.content or ""
            last_content = content
            reasoning = (
                getattr(message, "reasoning_content", None)
                or getattr(message, "reasoning", None)
                or ""
            )
            trace_round: dict = {
                "round": round_idx + 1,
                "reasoning": reasoning,
                "content": content,
                "global_time_offset": current_global_offset,
                "usage": getattr(response, "usage", None).model_dump() if getattr(response, "usage", None) else {},
            }
            if detailed:
                logger.info("[%s] Round %d content:\n%s", category_name, round_idx + 1, content)
                if reasoning:
                    logger.info("[%s] Round %d reasoning:\n%s", category_name, round_idx + 1, reasoning)

            # --- Extract tool calls: native first, XML fallback ---
            from tools.registry import parse_native_tool_calls
            native_calls = parse_native_tool_calls(message) if use_native_tools else []
            raw_tc_objects = getattr(message, "tool_calls", None) or []

            if native_calls:
                # Build assistant message preserving the tool_calls structure (needed for history)
                assistant_msg: Dict[str, Any] = {"role": "assistant", "content": content or ""}
                if raw_tc_objects:
                    assistant_msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                        for tc in raw_tc_objects
                    ]
                messages.append(assistant_msg)
                tool_calls = [c["arguments"] for c in native_calls]
                tool_call_ids = [tc.id for tc in raw_tc_objects]
                trace_round["tool_source"] = "native"
            else:
                content_clean = self._strip_thinking_tags(content)
                messages.append({"role": "assistant", "content": content_clean})
                tool_calls = self._extract_tool_calls(content)
                tool_call_ids = []
                if tool_calls:
                    trace_round["tool_source"] = "xml"

            if require_first_tool and not tool_used and round_idx == 0 and not tool_calls:
                trace_round["stop_reason"] = "first_round_requires_tool_only"
                trace["rounds"].append(trace_round)
                if not first_tool_reminder_sent:
                    reminder_text = self._get_complex_prompt("complex_first_round")
                    first_tool_reminder_sent = True
                else:
                    reminder_text = "请先用 clip_select 定位可疑时间窗，拿到切片后再输出JSON。"
                messages.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": reminder_text}],
                    }
                )
                continue

            parsed = ResponseParser.parse(category_name, content, retry_count=parse_attempt, is_abnormal=is_abnormal)
            if not ResponseParser.is_parse_failed(parsed):
                if require_first_tool and not tool_used:
                    trace_round["stop_reason"] = "json_parsed_but_tool_required"
                    trace["rounds"].append(trace_round)
                    messages.append(
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": "请先用 clip_select 定位可疑时间窗，拿到切片后再输出JSON。"}],
                        }
                    )
                    continue
                trace_round["stop_reason"] = "json_parsed"
                trace["rounds"].append(trace_round)
                trace["final_global_offset"] = current_global_offset
                return content, trace

            if not tool_calls:
                if require_first_tool and not tool_used:
                    trace_round["stop_reason"] = "first_tool_call_required"
                    trace["rounds"].append(trace_round)
                    messages.append(
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": "请先用 clip_select 定位可疑时间窗，拿到切片后再输出JSON。"}],
                        }
                    )
                    continue
                if not format_retry_injected and round_idx < max_turns - 1:
                    trace_round["stop_reason"] = "format_retry_injected"
                    trace["rounds"].append(trace_round)
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": ResponseParser.generate_retry_prompt(is_abnormal=is_abnormal, category_name=category_name),
                                }
                            ],
                        }
                    )
                    format_retry_injected = True
                    continue
                trace_round["stop_reason"] = "no_tool_call"
                trace["rounds"].append(trace_round)
                break

            video_path = self._video_url_to_local_path(video_url)
            duration = await self._ffprobe_video_duration(video_path)
            if duration > 0 and current_video_duration <= 0:
                current_video_duration = duration
            all_clip_infos: List[Dict[str, Any]] = []
            executed_calls: List[dict] = []
            rejected_global_calls: List[dict] = []

            for call_idx, call in enumerate(tool_calls):
                normalized = self._normalize_tool_arguments(
                    call,
                    duration,
                    category_name=category_name,
                    round_idx=effective_round_idx,
                )
                if (
                    category_complexity == "complex"
                    and not pre_sliced_input
                    and round_idx == 0
                    and self._is_global_window(normalized["start_time"], normalized["end_time"], duration)
                ):
                    rejected_global_calls.append(
                        {
                            "requested": call,
                            "normalized": normalized,
                            "reject_reason": "first_refine_must_be_local_window",
                        }
                    )
                    continue
                clip_info = await self._call_clip_select(
                    video_path,
                    normalized,
                    output_dir,
                    category_name=category_name,
                    round_idx=round_idx,
                    call_idx=call_idx,
                    global_time_offset=current_global_offset,
                )
                executed_calls.append(
                    {
                        "requested": call,
                        "normalized": normalized,
                        "global_start_time": clip_info.get("global_start_time", normalized.get("start_time")),
                        "global_end_time": clip_info.get("global_end_time", normalized.get("end_time")),
                        "clip_path": clip_info.get("clip_path", ""),
                        "saved_clip_path": clip_info.get("saved_clip_path", ""),
                    }
                )
                all_clip_infos.append(clip_info)
                if detailed:
                    logger.info(
                        "[%s] Round %d tool_call => %s | clip=%s",
                        category_name,
                        round_idx + 1,
                        json.dumps(normalized, ensure_ascii=False),
                        clip_info.get("clip_path", ""),
                    )

            if rejected_global_calls and not all_clip_infos:
                trace_round["stop_reason"] = "global_tool_call_rejected"
                trace_round["tool_calls_rejected"] = rejected_global_calls
                trace["rounds"].append(trace_round)
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": self._get_complex_prompt("complex_global_rejected"),
                            }
                        ],
                    }
                )
                continue

            if not all_clip_infos:
                trace_round["stop_reason"] = "tool_call_failed"
                trace_round["tool_calls"] = executed_calls
                trace["rounds"].append(trace_round)
                break

            trace_round["tool_calls"] = executed_calls
            trace_round["stop_reason"] = "tool_response_appended"
            trace["rounds"].append(trace_round)
            tool_used = True
            if all_clip_infos:
                first_info = all_clip_infos[0]
                try:
                    current_global_offset = float(first_info.get("global_start_time", current_global_offset))
                except (TypeError, ValueError):
                    pass
                clip_duration = first_info.get("end_time", 0.0) - first_info.get("start_time", 0.0)
                if clip_duration > 0:
                    current_video_duration = clip_duration
            if tool_call_ids:
                # Native: one tool message per call with matching tool_call_id
                for i, clip_info in enumerate(all_clip_infos):
                    tc_id = tool_call_ids[i] if i < len(tool_call_ids) else f"call_{i}"
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "content": self._build_tool_response_content(
                                category_name=category_name,
                                clip_infos=[clip_info],
                            ),
                        }
                    )
            else:
                # XML fallback: single tool message (no tool_call_id)
                messages.append(
                    {
                        "role": "tool",
                        "content": self._build_tool_response_content(
                            category_name=category_name,
                            clip_infos=all_clip_infos,
                        ),
                    }
                )

        trace["final_global_offset"] = current_global_offset
        return last_content, trace

    async def _async_call_api(
        self,
        messages: List[Dict[str, Any]],
        category_name: str,
        parse_attempt: int = 0,
        round_idx: int = 0,
        duration: float = 0.0,
        tools: List[Dict] | None = None,
        tool_choice: str | None = None,
    ):
        temperature = self.config["model"]["temperature"]
        if parse_attempt > 0:
            temperature = min(1.0, temperature + 0.1 * parse_attempt)
        fps = self._resolve_mm_processor_fps(category_name, round_idx, duration=duration)
        model_cfg = self.config["model"]
        extra_body: Dict[str, Any] = {
            "top_k": model_cfg["top_k"],
            "presence_penalty": model_cfg["presence_penalty"],
            "repetition_penalty": model_cfg["repetition_penalty"],
            "mm_processor_kwargs": {
                "fps": fps,
                "do_sample_frames": self.config["video"]["do_sample_frames"],
            },
        }
        if bool(model_cfg.get("enable_thinking", False)):
            extra_body["chat_template_kwargs"] = {"enable_thinking": True}

        kwargs: Dict[str, Any] = dict(
            model=self.config["api"]["model_name"],
            messages=messages,
            max_tokens=self.config["model"]["max_tokens"],
            temperature=temperature,
            top_p=self.config["model"]["top_p"],
            extra_body=extra_body,
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice or "auto"
        return await self.client.chat.completions.create(**kwargs)

    def _resolve_mm_processor_fps(self, category_name: str, round_idx: int, duration: float = 0.0) -> int | float:
        import math
        try:
            fps = float(self._resolve_video_fps(category_name))
        except (TypeError, ValueError):
            fps = 1.0
        if fps <= 0:
            fps = 1.0
        tool_cfg = self.config.get("tool_call", {})
        fixed_cfg = tool_cfg.get("fixed_sampling", {})
        if bool(fixed_cfg.get("enabled", False)):
            complexity = self._resolve_category_complexity(category_name)
            phase_key = self._resolve_sampling_phase_key(round_idx)
            profile_cfg = (
                fixed_cfg.get("profiles", {}).get(complexity, {}).get(phase_key, {})
                or fixed_cfg.get("profiles", {}).get(complexity, {}).get("later_round", {})
                or fixed_cfg.get("profiles", {}).get(complexity, {}).get("third_round", {})
                or fixed_cfg.get(phase_key, {})
            )
            fixed_fps = profile_cfg.get("fps")
            if fixed_fps is not None:
                try:
                    fixed_fps_val = float(fixed_fps)
                    if fixed_fps_val > 0:
                        fps = max(fps, fixed_fps_val)
                except (TypeError, ValueError):
                    pass
            fixed_interval = profile_cfg.get("sampling_interval")
            if fixed_interval is not None:
                try:
                    fixed_interval_val = float(fixed_interval)
                    if fixed_interval_val > 0:
                        fps = max(fps, 1.0 / fixed_interval_val)
                except (TypeError, ValueError):
                    pass
        return fps

    def _resolve_sampling_phase_key(self, round_idx: int) -> str:
        fixed_cfg = self.config.get("tool_call", {}).get("fixed_sampling", {})
        phases = fixed_cfg.get("phases_order")
        if isinstance(phases, list) and phases:
            idx = max(0, int(round_idx))
            if idx < len(phases):
                return str(phases[idx])
            return str(phases[-1])
        if round_idx <= 0:
            return "first_round"
        if round_idx == 1:
            return "second_round"
        return "third_round"

    def _resolve_video_fps(self, category_name: str) -> int | float:
        video_cfg = self.config.get("video", {})
        category_fps = video_cfg.get("category_fps", {})
        if category_name in category_fps:
            return category_fps[category_name]
        return video_cfg.get("fps", 1)

    def _resolve_category_complexity(self, category_name: str) -> str:
        pipeline_cfg = self.config.get("pipeline", {})
        complexity_map = pipeline_cfg.get("category_complexity", {})
        complexity = complexity_map.get(category_name, "simple")
        if isinstance(complexity, str):
            return complexity.lower()
        return "simple"

    def _is_global_window(self, start_time: float, end_time: float, duration: float) -> bool:
        if duration <= 0:
            return False
        window = max(0.0, end_time - start_time)
        return (window / duration) >= 0.8

    def _load_reference_previous(self) -> str:
        skills_dir = self.config.get("paths", {}).get("skills", "skills")
        try:
            return (Path(skills_dir) / "reference_previous.md").read_text(encoding="utf-8").strip()
        except Exception:
            return ""

    def _build_category_messages(
        self,
        video_url: str,
        category_prompt: str,
        category_name: str,
        previous_preds: dict,
        is_abnormal: bool = False,
    ) -> List[Dict[str, Any]]:
        # Choose system prompt: split prompts take priority over legacy global prompt
        if is_abnormal and self._complex_system_prompt:
            system_prompt = self._complex_system_prompt
        elif not is_abnormal and self._simple_system_prompt:
            system_prompt = self._simple_system_prompt
        else:
            system_prompt = self._system_prompt
        user_text_parts: list[str] = []
        if previous_preds:
            ref_prev = self._load_reference_previous()
            if ref_prev:
                system_prompt = system_prompt + "\n\n" + ref_prev
            prior_context = self._build_prior_context(previous_preds)
            if prior_context:
                user_text_parts.append(prior_context)
                user_text_parts.append("\n---\n")

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
        ]

        user_text_parts.append(f"当前任务大类：{category_name}\n{category_prompt}")

        user_content: List[Dict[str, Any]] = [
            {"type": "video_url", "video_url": {"url": video_url}},
            {"type": "text", "text": "\n".join(user_text_parts)},
        ]
        messages.append({"role": "user", "content": user_content})
        return messages

    def _build_prior_context(self, previous_preds: dict) -> str:
        label_map = {
            "自然时间段": "当前时间段",
            "人工光源辅助": "光照来源",
            "气象条件": "天气",
            "路面状态与视线干扰": "路面与视线",
            "主干道路级别": "道路级别",
            "特殊路段与设施": "道路设施",
            "静态障碍与全局异常": "全局交通状态",
            "弱势交通参与者异常": "弱势参与者风险",
            "车辆轨迹与空间冲突": "车辆轨迹冲突",
            "极端高危与失控事件": "极端高危事件",
        }
        ordered_categories = [
            "自然时间段",
            "人工光源辅助",
            "气象条件",
            "路面状态与视线干扰",
            "主干道路级别",
            "特殊路段与设施",
            "静态障碍与全局异常",
            "弱势交通参与者异常",
            "车辆轨迹与空间冲突",
            "极端高危与失控事件",
        ]
        lines: list[str] = []
        for cat in ordered_categories:
            pred = previous_preds.get(cat)
            if not isinstance(pred, list) or not pred:
                continue
            normalized = [str(item).strip() for item in pred if str(item).strip()]
            if not normalized:
                continue
            label = label_map.get(cat, cat)
            lines.append(f"- {label}: {', '.join(normalized)}")
        if not lines:
            return ""
        return (
            "先验上下文（用于收敛当前判断，若与当前视频可见证据冲突则以当前视频为准）：\n"
            + "\n".join(lines)
        )

    def _strip_thinking_tags(self, content: str) -> str:
        return re.sub(r"</?think>.*?</think>", "", content, flags=re.DOTALL).strip()

    def _extract_tool_calls(self, content: str) -> List[Dict[str, Any]]:
        """Extract clip_select arguments from XML <tool> tags (fallback when native tool calling unavailable)."""
        tool_calls_match = re.search(r"<tool>(.*?)</tool>", content, re.DOTALL)
        if not tool_calls_match:
            return []
        tool_calls_text = tool_calls_match.group(1).strip()
        raw_arguments = re.findall(
            r'{\s*"tool":\s*"(?:clip_select|frame_select)",\s*"(?:arguments|parameters)":\s*({.*?})\s*}',
            tool_calls_text,
            re.DOTALL,
        )
        if not raw_arguments:
            return []
        parsed: List[Dict[str, Any]] = []
        for item in raw_arguments:
            try:
                obj = json.loads(item.strip())
                if isinstance(obj, dict):
                    parsed.append(obj)
            except json.JSONDecodeError:
                continue
        return parsed

    def _video_url_to_local_path(self, video_url: str) -> str:
        prefix = str(self.config.get("paths", {}).get("video_url_prefix", "file:///app/videos")).rstrip("/") + "/"
        if not video_url.startswith(prefix):
            parsed = urlparse(video_url)
            if parsed.scheme == "file":
                return unquote(parsed.path)
            raise ValueError(f"不支持的视频URL: {video_url}")
        rel = video_url[len(prefix):]
        return str((Path(self.config["paths"]["video_root"]) / rel).resolve())

    async def _ffprobe_video_duration(self, video_path: str) -> float:
        proc = await asyncio.create_subprocess_exec(
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=duration",
            "-of",
            "json",
            video_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {stderr.decode().strip()}")
        data = json.loads(stdout.decode() or "{}")
        streams = data.get("streams", [])
        if not streams:
            return 0.0
        return float(streams[0].get("duration", 0.0) or 0.0)

    def _normalize_tool_arguments(
        self,
        arguments: Dict[str, Any],
        duration: float,
        category_name: str,
        round_idx: int,
    ) -> Dict[str, Any]:
        def safe_float(value, default: float) -> float:
            if value is None:
                return default
            try:
                return float(value)
            except (ValueError, TypeError):
                return default

        start_time = safe_float(arguments.get("start_time"), 0.0)
        end_time_val = arguments.get("end_time")
        if end_time_val is None:
            end_time = duration if duration > 0 else start_time + 8.0
        else:
            end_time = safe_float(end_time_val, duration if duration > 0 else start_time + 8.0)
        if duration > 0:
            start_time = max(0.0, min(start_time, duration))
            end_time = max(start_time + 0.1, min(end_time, duration))
        else:
            end_time = max(start_time + 0.1, end_time)

        sampling_interval_value = self._resolve_clip_sampling_interval(category_name, round_idx)
        fixed_applied = False
        tool_cfg = self.config.get("tool_call", {})
        fixed_cfg = tool_cfg.get("fixed_sampling", {})
        if bool(fixed_cfg.get("enabled", False)):
            complexity = self._resolve_category_complexity(category_name)
            phase_key = self._resolve_sampling_phase_key(round_idx)
            profile_cfg = (
                fixed_cfg.get("profiles", {}).get(complexity, {}).get(phase_key, {})
                or fixed_cfg.get("profiles", {}).get(complexity, {}).get("later_round", {})
                or fixed_cfg.get("profiles", {}).get(complexity, {}).get("third_round", {})
                or fixed_cfg.get(phase_key, {})
            )
            fixed_window = safe_float(profile_cfg.get("window_seconds"), 0.0)
            if fixed_window > 0:
                center_time = (start_time + end_time) / 2.0
                start_time, end_time = self._center_window(
                    center_time=center_time,
                    window_seconds=fixed_window,
                    duration=duration,
                )
                fixed_applied = True
        resize = safe_float(arguments.get("resize"), 1.0)
        return {
            "start_time": start_time,
            "end_time": end_time,
            "resize": resize,
            "sampling_interval": sampling_interval_value,
            "fixed_sampling_applied": fixed_applied,
        }

    def _center_window(
        self,
        center_time: float,
        window_seconds: float,
        duration: float,
    ) -> Tuple[float, float]:
        w = max(0.1, float(window_seconds))
        if duration > 0:
            w = min(w, duration)
        left = max(0.0, center_time - w / 2.0)
        right = min(duration, left + w) if duration > 0 else left + w
        left = max(0.0, right - w)
        return left, max(left + 0.1, right)

    def _resolve_clip_sampling_interval(self, category_name: str, round_idx: int) -> float:
        def _safe_float(value, default: float) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        video_cfg = self.config.get("video", {})
        default_fps = _safe_float(
            video_cfg.get("category_fps", {}).get(category_name, video_cfg.get("fps", 1)),
            1.0,
        )
        default_fps = max(default_fps, 1e-6)
        interval = 1.0 / default_fps

        tool_cfg = self.config.get("tool_call", {})
        fixed_cfg = tool_cfg.get("fixed_sampling", {})
        if bool(fixed_cfg.get("enabled", False)):
            complexity = self._resolve_category_complexity(category_name)
            phase_key = self._resolve_sampling_phase_key(round_idx)
            profile_cfg = (
                fixed_cfg.get("profiles", {}).get(complexity, {}).get(phase_key, {})
                or fixed_cfg.get("profiles", {}).get(complexity, {}).get("later_round", {})
                or fixed_cfg.get("profiles", {}).get(complexity, {}).get("third_round", {})
                or fixed_cfg.get(phase_key, {})
            )
            fixed_fps = _safe_float(profile_cfg.get("fps"), 0.0)
            if fixed_fps > 0:
                interval = 1.0 / max(default_fps, fixed_fps)
            else:
                fixed_interval = _safe_float(profile_cfg.get("sampling_interval"), 0.0)
                if fixed_interval > 0:
                    interval = min(interval, fixed_interval)

        return max(0.05, interval)

    async def _call_clip_select(
        self,
        video_path: str,
        arguments: Dict[str, Any],
        output_dir: str,
        category_name: str,
        round_idx: int,
        call_idx: int,
        save_name_prefix: str = "",
        global_time_offset: float = 0.0,
    ) -> Dict[str, Any]:
        tool_path = self.config.get("tool_call", {}).get("clip_tool_path", "select_clip_fallback.py")
        paths_cfg = self.config.get("paths", {})
        clip_save_root = str(
            Path(
                paths_cfg.get("tool_clip_root")
                or (Path(paths_cfg.get("output_base", "/app/outputs/scene_mining")) / "_tool_clips")
            ).resolve()
        )
        cmd_args = [
            "--video-path",
            video_path,
            "--start-time",
            str(arguments["start_time"]),
            "--end-time",
            str(arguments["end_time"]),
            "--sampling-interval",
            str(arguments["sampling_interval"]),
            "--save-root",
            clip_save_root,
            "--clamp",
        ]
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            tool_path,
            *cmd_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        stdout_text = stdout.decode().strip()
        if proc.returncode != 0:
            raise RuntimeError(f"clip_select失败: {stderr.decode().strip()}")
        match = re.search(
            r"Clip saved to (.*?), start=([0-9.]+), end=([0-9.]+), sampling_interval=([0-9.]+)",
            stdout_text,
        )
        if not match:
            raise RuntimeError(f"clip_select输出无法解析: {stdout_text}")
        host_clip_path = match.group(1).strip()
        clip_path = self._to_container_media_path(host_clip_path)
        start_time = float(match.group(2))
        end_time = float(match.group(3))
        sampling_interval = float(match.group(4))
        logger.info("clip_select generated clip: host=%s, media=%s", host_clip_path, clip_path)
        global_start_time = max(0.0, start_time + float(global_time_offset))
        global_end_time = max(global_start_time, end_time + float(global_time_offset))

        save_clip_cfg = bool(self.config.get("video", {}).get("save_sampled_clips", True))
        saved_clip_path = ""
        if save_clip_cfg:
            import shutil
            clips_output_dir = Path(output_dir) / "sampled_clips" / category_name
            clips_output_dir.mkdir(parents=True, exist_ok=True)
            src = Path(host_clip_path)
            suffix = src.suffix if src.suffix else ".mp4"
            start_tag = f"{start_time:.2f}".replace(".", "s")
            end_tag = f"{end_time:.2f}".replace(".", "s")
            global_start_tag = f"{global_start_time:.2f}".replace(".", "s")
            global_end_tag = f"{global_end_time:.2f}".replace(".", "s")
            interval_tag = f"{sampling_interval:.2f}".replace(".", "s")
            if save_name_prefix:
                safe_prefix = (
                    str(save_name_prefix)
                    .replace("/", "_")
                    .replace("\\", "_")
                    .replace(" ", "_")
                )
                filename = (
                    f"{safe_prefix}_g{global_start_tag}_{global_end_tag}_"
                    f"l{start_tag}_{end_tag}_{interval_tag}{suffix}"
                )
            else:
                filename = (
                    f"round{round_idx + 1}_call{call_idx + 1}_g{global_start_tag}_{global_end_tag}_"
                    f"l{start_tag}_{end_tag}_{interval_tag}{suffix}"
                )
            dst = clips_output_dir / filename
            shutil.copy2(src, dst)
            saved_clip_path = str(dst)

        return {
            "clip_path": clip_path,
            "host_clip_path": host_clip_path,
            "start_time": start_time,
            "end_time": end_time,
            "global_start_time": global_start_time,
            "global_end_time": global_end_time,
            "sampling_interval": sampling_interval,
            "saved_clip_path": saved_clip_path,
        }

    def _to_container_media_path(self, host_path: str) -> str:
        video_root = Path(self.config["paths"]["video_root"]).resolve()
        host = Path(host_path).resolve()
        try:
            rel = host.relative_to(video_root).as_posix()
        except ValueError:
            return str(host)
        prefix = str(self.config.get("paths", {}).get("video_url_prefix", "file:///app/videos")).rstrip("/")
        if prefix.startswith("file://"):
            return str(Path(unquote(urlparse(prefix).path)) / rel)
        return f"/app/videos/{rel}"

    def _build_tool_response_content(
        self,
        category_name: str,
        clip_infos: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = [{"type": "text", "text": ""}]
        for i, clip in enumerate(clip_infos):
            prefix = "\n" if i > 0 else ""
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"{prefix}Clip {i} [{clip['start_time']:.2f}s - {clip['end_time']:.2f}s], "
                        f"sampling_interval={clip['sampling_interval']:.2f}s: "
                    ),
                }
            )
            content.append({"type": "video_url", "video_url": {"url": f"file://{clip['clip_path']}"}})
        content.append(
            {
                "type": "text",
                "text": (
                    "\n当前任务「"
                    + category_name
                    + "」：证据充分则直接输出JSON，否则继续clip_select局部切片。"
                ),
            }
        )
        return content

    async def run_react_session(
        self,
        video_url: str,
        task_description: str,
        tool_registry,
        available_tool_names: list[str],
        context: dict,
        max_turns: int = 10,
        on_step=None,
    ) -> tuple[str, list[dict], dict]:
        """Run a ReAct think→act→observe loop with multi-tool support.

        Args:
            video_url: Video URL to analyze
            task_description: Category prompt / task text
            tool_registry: ToolRegistry instance
            available_tool_names: Which tools this session can use
            context: Shared context dict (video_path, config, etc.)
            max_turns: Max ReAct loop iterations
            on_step: Callback(step_type, data) for interactive output

        Returns:
            (final_content, conversation_history, trace_dict)
        """
        from tools.registry import parse_native_tool_calls, parse_xml_tool_calls

        use_native_tools = bool(
            self.config.get("react_agent", {}).get("use_native_tools", True)
        )

        # Build system prompt
        system_prompt = self._system_prompt
        if not use_native_tools:
            xml_desc = tool_registry.get_xml_tool_descriptions(available_tool_names)
            if xml_desc:
                system_prompt = system_prompt + "\n\n" + xml_desc

        # Build initial messages
        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
        ]

        user_content = [
            {"type": "video_url", "video_url": {"url": video_url}},
            {"type": "text", "text": task_description},
        ]
        messages.append({"role": "user", "content": user_content})

        trace: dict = {"rounds": [], "tool_calls": []}
        conversation: list[dict] = list(messages)
        last_content = ""

        for turn_idx in range(max_turns):
            # --- Call API ---
            api_kwargs: dict = {
                "model": self.config["api"]["model_name"],
                "messages": messages,
                "max_tokens": self.config["model"]["max_tokens"],
                "temperature": self.config["model"]["temperature"],
                "top_p": self.config["model"]["top_p"],
            }

            # Add native tools parameter if supported
            if use_native_tools:
                openai_tools = tool_registry.get_openai_tools_param(available_tool_names)
                if openai_tools:
                    api_kwargs["tools"] = openai_tools
                    api_kwargs["tool_choice"] = "auto"

            extra_body: dict = {
                "top_k": self.config["model"]["top_k"],
                "presence_penalty": self.config["model"]["presence_penalty"],
                "repetition_penalty": self.config["model"]["repetition_penalty"],
                "mm_processor_kwargs": {
                    "fps": self._resolve_video_fps_for_react(),
                    "do_sample_frames": self.config["video"].get("do_sample_frames", True),
                },
            }
            if bool(self.config["model"].get("enable_thinking", False)):
                extra_body["chat_template_kwargs"] = {"enable_thinking": True}
            api_kwargs["extra_body"] = extra_body

            try:
                response = await self.client.chat.completions.create(**api_kwargs)
            except Exception as exc:
                err_msg = str(exc)
                # If native tools fail (e.g. vLLM doesn't support --enable-auto-tool-choice),
                # auto-disable native tools for all subsequent turns
                if "tool choice" in err_msg.lower() or "tool-call-parser" in err_msg.lower() or "tool_choice" in err_msg.lower():
                    logger.warning(
                        "Native tools not supported by this model server, switching to XML fallback mode"
                    )
                    use_native_tools = False
                    # Rebuild system prompt with XML tool descriptions
                    xml_desc = tool_registry.get_xml_tool_descriptions(available_tool_names)
                    if xml_desc:
                        messages[0]["content"] = system_prompt + "\n\n" + xml_desc
                    # Retry this turn without native tools
                    retry_kwargs = {k: v for k, v in api_kwargs.items() if k not in ("tools", "tool_choice")}
                    try:
                        response = await self.client.chat.completions.create(**retry_kwargs)
                    except Exception as exc2:
                        logger.error("ReAct API call failed at turn %d (after tool fallback): %s", turn_idx + 1, exc2)
                        trace["error"] = str(exc2)
                        break
                else:
                    logger.error("ReAct API call failed at turn %d: %s", turn_idx + 1, exc)
                    trace["error"] = str(exc)
                    break

            message = response.choices[0].message
            content = message.content or ""
            reasoning = (
                getattr(message, "reasoning_content", None)
                or getattr(message, "reasoning", None)
                or ""
            )
            last_content = content

            trace_round: dict = {
                "turn": turn_idx + 1,
                "reasoning": reasoning,
                "content": content,
                "usage": getattr(response, "usage", None).model_dump() if getattr(response, "usage", None) else {},
            }

            # --- Notify callback ---
            if on_step:
                try:
                    if reasoning:
                        on_step("thinking", {"content": reasoning[:500]})
                    if content and not reasoning:
                        on_step("thinking", {"content": content[:300]})
                except Exception:
                    pass

            # --- Strip thinking tags from content ---
            content_clean = self._strip_thinking_tags(content)
            messages.append({"role": "assistant", "content": content_clean})

            # --- Try to extract tool calls ---
            tool_calls_parsed = []

            # Strategy 1: Native function calling (always try, regardless of use_native_tools setting)
            native_calls = parse_native_tool_calls(message)
            if native_calls:
                tool_calls_parsed = native_calls

            # Strategy 2: XML fallback
            if not tool_calls_parsed:
                xml_calls = parse_xml_tool_calls(content)
                if xml_calls:
                    tool_calls_parsed = xml_calls

            # --- Check for final JSON result ---
            parsed_result = ResponseParser.parse(
                "", content, is_abnormal=True  # generic parse
            )
            has_valid_json = not ResponseParser.is_parse_failed(parsed_result)

            if tool_calls_parsed:
                # Execute tool calls
                all_tool_results = []
                executed_calls = []

                for call in tool_calls_parsed:
                    tool_name = call.get("tool", "")
                    tool_args = call.get("arguments", {})

                    if on_step:
                        try:
                            on_step("tool_call", {"tool": tool_name, "args": tool_args})
                        except Exception:
                            pass

                    result = await tool_registry.dispatch(tool_name, tool_args, context)
                    executed_calls.append({
                        "tool": tool_name,
                        "arguments": tool_args,
                        "output": result.output[:200],
                    })
                    trace["tool_calls"].append({
                        "turn": turn_idx + 1,
                        "tool": tool_name,
                        "arguments": tool_args,
                        "output_preview": result.output[:200],
                    })
                    all_tool_results.append(result)

                    if on_step:
                        try:
                            on_step("tool_result", {
                                "tool": tool_name,
                                "output": result.output[:300],
                            })
                        except Exception:
                            pass

                trace_round["tool_calls"] = executed_calls
                trace_round["stop_reason"] = "tool_calls_executed"
                trace["rounds"].append(trace_round)

                # Build tool response message
                tool_response_parts: list[dict] = []
                for tr in all_tool_results:
                    tool_response_parts.append({
                        "type": "text",
                        "text": tr.output,
                    })
                    if tr.media_blocks:
                        tool_response_parts.extend(tr.media_blocks)

                # Append category prompt reminder
                tool_response_parts.append({
                    "type": "text",
                    "text": f"\n请继续分析。当前任务：{task_description[:200]}\n"
                            "证据充分则直接输出JSON结果，不足则继续调用工具。",
                })

                messages.append({"role": "tool", "content": tool_response_parts})
                continue

            elif has_valid_json and not tool_calls_parsed:
                # Valid JSON result, no more tool calls — we're done
                trace_round["stop_reason"] = "json_parsed"
                trace["rounds"].append(trace_round)

                if on_step:
                    try:
                        on_step("final", {"result": parsed_result})
                    except Exception:
                        pass

                return content, conversation, trace

            else:
                # No tool calls, no valid JSON — inject format correction
                trace_round["stop_reason"] = "no_tool_no_json"
                trace["rounds"].append(trace_round)

                retry_prompt = (
                    "请输出有效的JSON结果。如果需要更多信息，请调用工具。"
                    "不要输出其他文本，只输出JSON或工具调用。"
                )
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": retry_prompt}],
                })
                continue

        # Max turns reached
        trace["max_turns_reached"] = True
        return last_content, conversation, trace

    def _resolve_video_fps_for_react(self) -> int | float:
        """Resolve FPS for ReAct session API calls."""
        video_cfg = self.config.get("video", {})
        fps = float(video_cfg.get("fps", 1))
        if fps <= 0:
            fps = 1.0
        # For ReAct, default to a moderate FPS for the initial video view
        return fps

    async def close(self) -> None:
        await self.http_client.aclose()
