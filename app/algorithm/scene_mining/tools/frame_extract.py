"""
frame_extract tool — extract specific frames at high resolution.

Uses ffmpeg to extract individual frames as JPEG images for detailed inspection.
"""

import asyncio
import logging
from pathlib import Path

from tools.registry import ToolDefinition, ToolResult

logger = logging.getLogger(__name__)

MAX_FRAMES = 5


async def execute_frame_extract(args: dict, context: dict) -> ToolResult:
    """Execute frame_extract: extract specific frames at high resolution."""
    config = context.get("config", {})
    video_path = context.get("video_path", "")
    output_dir = context.get("output_dir", "")

    timestamps = args.get("timestamps", [])
    if not isinstance(timestamps, list):
        timestamps = [timestamps]
    timestamps = [float(t) for t in timestamps if isinstance(t, (int, float))]
    timestamps = sorted(set(timestamps))[:MAX_FRAMES]

    if not timestamps:
        return ToolResult(output="错误：未提供有效的时间戳列表")

    # Create output directory for frames
    frames_dir = Path(output_dir) / "extracted_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    media_blocks = []
    frame_paths = []

    for i, ts in enumerate(timestamps):
        output_path = frames_dir / f"frame_{ts:.2f}s.jpg"
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(ts),
            "-i", video_path,
            "-frames:v", "1",
            "-q:v", "2",
            str(output_path),
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            logger.warning("ffmpeg frame extract failed at %.2fs: %s", ts, stderr.decode().strip()[:200])
            continue

        if output_path.exists():
            # Build image URL for model input
            frame_url = f"file://{output_path.resolve()}"
            media_blocks.append({
                "type": "image_url",
                "image_url": {"url": frame_url},
            })
            frame_paths.append(str(output_path))

    if not frame_paths:
        return ToolResult(output="帧提取失败：所有时间戳均提取失败")

    output_text = f"已提取 {len(frame_paths)} 帧高分辨率图像："
    for i, ts in enumerate(timestamps[:len(frame_paths)]):
        output_text += f"\n  - {ts:.2f}s"

    return ToolResult(
        output=output_text,
        media_blocks=media_blocks,
        metadata={"frame_paths": frame_paths, "timestamps": timestamps[:len(frame_paths)]},
    )


FRAME_EXTRACT_TOOL = ToolDefinition(
    name="frame_extract",
    description=(
        "提取指定时间点的高分辨率视频帧，用于仔细检查特定时刻的细节。"
        "最多提取5帧。适用于需要看清目标外观、路标文字、信号灯状态等场景。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "timestamps": {
                "type": "array",
                "items": {"type": "number"},
                "description": "要提取的时间点列表（秒），如 [3.5, 7.2, 12.0]，最多5个",
            },
        },
        "required": ["timestamps"],
    },
    executor=execute_frame_extract,
)
