"""
yolo_detect tool — run YOLO object detection on a time window.

Wraps the YOLO pipeline from nodes/yolo_pre_filter.py for on-demand detection.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from tools.registry import ToolDefinition, ToolResult

logger = logging.getLogger(__name__)

# Reuse model cache from yolo_pre_filter
from nodes.yolo_pre_filter import (
    _load_yolo_model,
    _sample_frames,
    _run_yolo_inference,
    _parse_yolo_result,
    COCO_NAMES,
    DEFAULT_CATEGORY_CLASSES,
)


async def execute_yolo_detect(args: dict, context: dict) -> ToolResult:
    """Execute yolo_detect: run YOLO on a time window."""
    config = context.get("config", {})
    video_path = context.get("video_path", "")
    output_dir = context.get("output_dir", "")
    video_info = context.get("video_info", {})
    duration = float(video_info.get("duration", 0) or 0)

    yolo_cfg = config.get("yolo", {})
    if not yolo_cfg:
        return ToolResult(output="YOLO 未配置，无法执行目标检测")

    start_time = float(args.get("start_time", 0))
    end_time = float(args.get("end_time", duration))
    conf_threshold = float(args.get("conf_threshold", yolo_cfg.get("conf_threshold", 0.35)))

    # If a time window is specified and it's not the full video, extract a clip first
    is_partial_window = (end_time - start_time) < duration * 0.9 and duration > 0
    clip_path = video_path

    if is_partial_window:
        # Use clip_select tool to extract the window first
        from tools.clip_select import execute_clip_select
        clip_result = await execute_clip_select(
            {
                "start_time": start_time,
                "end_time": end_time,
                "sampling_level": "scan",
            },
            context,
        )
        if clip_result.metadata and "clip_path" in clip_result.metadata:
            # Resolve clip path to local filesystem
            container_path = clip_result.metadata["clip_path"]
            video_root = Path(config.get("paths", {}).get("video_root", "")).resolve()
            clip_path = str(video_root / container_path.replace("/app/videos/", ""))
        actual_duration = end_time - start_time
    else:
        actual_duration = duration

    # Run YOLO in thread pool to avoid blocking
    try:
        result_data = await asyncio.to_thread(
            _run_yolo_detect_sync,
            clip_path, actual_duration, conf_threshold, yolo_cfg, config,
        )
    except Exception as e:
        logger.error("YOLO detect error: %s", e)
        return ToolResult(output=f"YOLO 检测失败: {e}")

    if not result_data:
        return ToolResult(
            output=f"在 {start_time:.1f}s-{end_time:.1f}s 时间段内未检测到目标"
        )

    # Format results as readable text
    lines = [f"YOLO 检测结果 ({start_time:.1f}s - {end_time:.1f}s)："]
    for category, dets in result_data.items():
        if dets:
            lines.append(f"\n  {category}：")
            for det in dets:
                lines.append(
                    f"    - {det['class']} (置信度={det['confidence']:.2f}, "
                    f"时间={det.get('timestamp', '?')}s)"
                )
    lines.append(f"\n  共检测到 {sum(len(d) for d in result_data.values())} 个目标")

    return ToolResult(
        output="\n".join(lines),
        metadata={"detections": result_data, "start_time": start_time, "end_time": end_time},
    )


def _run_yolo_detect_sync(
    video_path: str,
    duration: float,
    conf_threshold: float,
    yolo_cfg: dict,
    config: dict,
) -> dict[str, list[dict]]:
    """Run YOLO detection synchronously and return per-category detections."""
    model_path = str(yolo_cfg.get("model_path", ""))
    model_path_fallback = str(yolo_cfg.get("model_path_fallback", ""))
    device = str(yolo_cfg.get("device", "0"))
    sample_fps = float(yolo_cfg.get("sample_fps", 4.0))
    iou_threshold = float(yolo_cfg.get("iou_threshold", 0.45))
    imgsz = int(yolo_cfg.get("imgsz", 640))
    batch_size = int(yolo_cfg.get("batch_size", 64))
    half = bool(yolo_cfg.get("half", True))

    category_classes = yolo_cfg.get("category_classes", DEFAULT_CATEGORY_CLASSES)

    # Load model
    model = None
    if Path(model_path).exists():
        model = _load_yolo_model(model_path, device)
    elif model_path_fallback and Path(model_path_fallback).exists():
        model = _load_yolo_model(model_path_fallback, device)

    if model is None:
        logger.error("No YOLO model found")
        return {}

    # Sample frames
    frames, _ = _sample_frames(video_path, sample_fps, duration)
    if not frames:
        return {}

    # Run inference
    frame_results = _run_yolo_inference(
        model, frames, conf_threshold, iou_threshold, imgsz, device, half,
        batch_size, category_classes,
    )

    # Collect per-category detections
    result: dict[str, list[dict]] = {cat: [] for cat in category_classes}
    for fr in frame_results:
        ts = fr.get("timestamp", 0)
        for category, dets in fr.get("category_detections", {}).items():
            for det in dets:
                result.setdefault(category, []).append({
                    "class": det["class"],
                    "confidence": det["confidence"],
                    "timestamp": round(ts, 2),
                })

    return result


YOLO_DETECT_TOOL = ToolDefinition(
    name="yolo_detect",
    description=(
        "对指定时间段运行 YOLO 目标检测，快速识别行人、车辆、自行车等目标。"
        "适用于需要快速确认某区域是否存在特定目标的场景。"
        "注意：此工具仅返回目标检测结果，不做行为分析。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "start_time": {
                "type": "number",
                "description": "检测起始时间（秒）",
            },
            "end_time": {
                "type": "number",
                "description": "检测结束时间（秒）",
            },
            "conf_threshold": {
                "type": "number",
                "description": "置信度阈值，默认0.35",
                "default": 0.35,
            },
        },
        "required": ["start_time", "end_time"],
    },
    executor=execute_yolo_detect,
)
