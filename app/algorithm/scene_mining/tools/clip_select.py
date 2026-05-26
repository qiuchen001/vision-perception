"""
clip_select tool — extract a time window from video.

Extracted from qwen_client.py for the ReAct agent tool system.
Supports progressive sampling via sampling_level parameter.
"""

import asyncio
import json
import logging
import math
import re
import shutil
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse
from typing import Any, Dict, Tuple

from tools.registry import ToolDefinition, ToolResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Progressive sampling levels
# ---------------------------------------------------------------------------

SAMPLING_LEVELS = {
    "overview":  {"fps": 1,  "window_seconds": 30, "description": "低精度概览，用于形成初步假设"},
    "scan":      {"fps": 4,  "window_seconds": 10, "description": "中精度扫描，验证可疑区域"},
    "focus":     {"fps": 8,  "window_seconds": 6,  "description": "高精度聚焦，观察目标和行为"},
    "pinpoint":  {"fps": 16, "window_seconds": 2,  "description": "极高精度定位，精确定位事件边界"},
}


def resolve_sampling_from_level(
    sampling_level: str | None,
    default_fps: float,
    default_window: float,
    duration: float,
    config: dict,
) -> tuple[float, float]:
    """Resolve FPS and window_seconds from sampling_level or config fallback.

    Returns (fps, window_seconds).
    """
    # Try sampling_level first
    if sampling_level and sampling_level in SAMPLING_LEVELS:
        level_cfg = SAMPLING_LEVELS[sampling_level]
        fps = float(level_cfg["fps"])
        window = float(level_cfg["window_seconds"])
    else:
        # Fallback to config fixed_sampling
        tool_cfg = config.get("tool_call", {})
        fixed_cfg = tool_cfg.get("fixed_sampling", {})
        if bool(fixed_cfg.get("enabled", False)):
            # Use first_round defaults
            profile = (
                fixed_cfg.get("profiles", {}).get("complex", {}).get("first_round", {})
            )
            fps = float(profile.get("fps", default_fps))
            window = float(profile.get("window_seconds", default_window))
        else:
            fps = default_fps
            window = default_window

    return fps, window


# ---------------------------------------------------------------------------
# Argument normalization (from qwen_client.py)
# ---------------------------------------------------------------------------

def _center_window(
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


def normalize_clip_arguments(
    arguments: dict,
    duration: float,
    sampling_level: str | None = None,
    config: dict | None = None,
) -> dict:
    """Normalize clip_select arguments: clamp times, resolve FPS/window."""
    config = config or {}

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

    # Resolve FPS and window from sampling_level or config
    default_fps = safe_float(
        config.get("video", {}).get("fps", 1), 1.0
    )
    default_window = 8.0
    fps, window_seconds = resolve_sampling_from_level(
        sampling_level, default_fps, default_window, duration, config,
    )

    # Apply window clamping from sampling level
    if sampling_level and sampling_level in SAMPLING_LEVELS and window_seconds > 0:
        center = (start_time + end_time) / 2.0
        start_time, end_time = _center_window(
            center_time=center,
            window_seconds=window_seconds,
            duration=duration,
        )
    elif duration > 0:
        start_time = max(0.0, min(start_time, duration))
        end_time = max(start_time + 0.1, min(end_time, duration))
    else:
        end_time = max(start_time + 0.1, end_time)

    # Compute sampling_interval from fps
    sampling_interval = max(0.05, 1.0 / max(fps, 1e-6))

    resize = safe_float(arguments.get("resize"), 1.0)
    return {
        "start_time": start_time,
        "end_time": end_time,
        "resize": resize,
        "sampling_interval": sampling_interval,
        "fps": fps,
        "sampling_level": sampling_level or "scan",
    }


# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------

async def execute_clip_select(args: dict, context: dict) -> ToolResult:
    """Execute clip_select: extract a time window from video via ffmpeg."""
    config = context.get("config", {})
    video_path = context.get("video_path", "")
    output_dir = context.get("output_dir", "")
    video_info = context.get("video_info", {})
    duration = float(video_info.get("duration", 0) or 0)

    sampling_level = args.get("sampling_level") or args.get("sampling_level")
    normalized = normalize_clip_arguments(args, duration, sampling_level, config)

    tool_path = config.get("tool_call", {}).get("clip_tool_path", "select_clip_fallback.py")
    clip_save_root = str(Path(config.get("paths", {}).get("video_root", "")) / "_tool_clips")

    cmd_args = [
        "--video-path", video_path,
        "--start-time", str(normalized["start_time"]),
        "--end-time", str(normalized["end_time"]),
        "--sampling-interval", str(normalized["sampling_interval"]),
        "--save-root", clip_save_root,
        "--clamp",
    ]
    proc = await asyncio.create_subprocess_exec(
        sys.executable, tool_path, *cmd_args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    stdout_text = stdout.decode().strip()

    if proc.returncode != 0:
        return ToolResult(output=f"clip_select 执行失败: {stderr.decode().strip()}")

    match = re.search(
        r"Clip saved to (.*?), start=([0-9.]+), end=([0-9.]+), sampling_interval=([0-9.]+)",
        stdout_text,
    )
    if not match:
        return ToolResult(output=f"clip_select 输出无法解析: {stdout_text}")

    host_clip_path = match.group(1).strip()
    clip_path = _to_container_media_path(host_clip_path, config)
    start_time = float(match.group(2))
    end_time = float(match.group(3))
    sampling_interval = float(match.group(4))

    # Save clip to output dir for debugging
    saved_clip_path = ""
    save_clip_cfg = bool(config.get("video", {}).get("save_sampled_clips", True))
    if save_clip_cfg and output_dir:
        import shutil as sh
        clips_output_dir = Path(output_dir) / "sampled_clips" / "clip_select"
        clips_output_dir.mkdir(parents=True, exist_ok=True)
        src = Path(host_clip_path)
        suffix = src.suffix if src.suffix else ".mp4"
        start_tag = f"{start_time:.2f}".replace(".", "s")
        end_tag = f"{end_time:.2f}".replace(".", "s")
        filename = f"clip_{start_tag}_{end_tag}{suffix}"
        dst = clips_output_dir / filename
        sh.copy2(src, dst)
        saved_clip_path = str(dst)

    # Build video_url content block
    clip_url = f"file://{clip_path}"
    media_blocks = [
        {"type": "video_url", "video_url": {"url": clip_url}}
    ]

    level_str = f" (级别: {sampling_level})" if sampling_level else ""
    output_text = (
        f"视频切片已提取{level_str}: {start_time:.2f}s - {end_time:.2f}s, "
        f"采样FPS={normalized['fps']}, 采样间隔={sampling_interval:.2f}s"
    )

    return ToolResult(
        output=output_text,
        media_blocks=media_blocks,
        metadata={
            "clip_path": clip_path,
            "start_time": start_time,
            "end_time": end_time,
            "fps": normalized["fps"],
            "sampling_interval": sampling_interval,
            "sampling_level": sampling_level or "scan",
            "saved_clip_path": saved_clip_path,
        },
    )


def _to_container_media_path(host_path: str, config: dict) -> str:
    video_root = Path(config.get("paths", {}).get("video_root", "")).resolve()
    host = Path(host_path).resolve()
    try:
        rel = host.relative_to(video_root).as_posix()
    except ValueError:
        rel = host.name
    prefix = str(config.get("paths", {}).get("video_url_prefix", "file:///app/videos")).rstrip("/")
    if prefix.startswith("file://"):
        return str(Path(unquote(urlparse(prefix).path)) / rel)
    return f"/app/videos/{rel}"


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

CLIP_SELECT_TOOL = ToolDefinition(
    name="clip_select",
    description=(
        "提取视频指定时间段的切片，用于仔细观察可疑区域。"
        "支持渐进采样：overview=低精度概览, scan=中精度扫描, "
        "focus=高精度聚焦, pinpoint=极高精度定位。"
        "首次观察用 overview/scan，确认异常后用 focus/pinpoint 精确定位。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "start_time": {
                "type": "number",
                "description": "起始时间（秒），≥0",
            },
            "end_time": {
                "type": "number",
                "description": "结束时间（秒），>start_time",
            },
            "sampling_level": {
                "type": "string",
                "enum": ["overview", "scan", "focus", "pinpoint"],
                "default": "scan",
                "description": "采样精度级别：overview=低精度概览(1fps/30s窗), scan=中精度扫描(4fps/10s窗), focus=高精度聚焦(8fps/6s窗), pinpoint=极高精度定位(16fps/2s窗)",
            },
        },
        "required": ["start_time", "end_time"],
    },
    executor=execute_clip_select,
)
