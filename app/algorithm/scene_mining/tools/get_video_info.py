"""
get_video_info tool — return video metadata.

Extracted from langgraph_pipeline.py for the ReAct agent tool system.
"""

import asyncio
import json
import logging
from pathlib import Path

from tools.registry import ToolDefinition, ToolResult

logger = logging.getLogger(__name__)


async def execute_get_video_info(args: dict, context: dict) -> ToolResult:
    """Execute get_video_info: return video duration, fps, resolution."""
    video_path = context.get("video_path", "")
    video_info = context.get("video_info", {})

    duration = float(video_info.get("duration", 0) or 0)
    fps = int(video_info.get("fps", 0) or 0)

    # Try to get resolution via ffprobe
    resolution = "unknown"
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "json",
            video_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode == 0:
            data = json.loads(stdout.decode() or "{}")
            streams = data.get("streams", [])
            if streams:
                w = streams[0].get("width", "?")
                h = streams[0].get("height", "?")
                resolution = f"{w}x{h}"
    except Exception:
        pass

    result = {
        "duration": f"{duration:.1f}s",
        "fps": fps,
        "resolution": resolution,
        "video_path": video_path,
    }
    return ToolResult(
        output=json.dumps(result, ensure_ascii=False, indent=2),
        metadata=result,
    )


GET_VIDEO_INFO_TOOL = ToolDefinition(
    name="get_video_info",
    description="获取当前视频的基本信息：时长、帧率、分辨率。",
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    executor=execute_get_video_info,
)
