"""
YOLO Pre-Filter Node for LangGraph pipeline.

Runs YOLO object detection on sampled frames to identify suspicious time windows
for complex categories, replacing the LLM full-video scan in supervisor_2.

Architecture: One YOLO pass → results mapped to 3 complex categories → time windows.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model singleton
# ---------------------------------------------------------------------------
_model_cache: dict[str, Any] = {}


def _load_yolo_model(model_path: str, device: str) -> Any:
    """Load YOLO model from checkpoint path (singleton per path)."""
    from ultralytics import YOLO

    if model_path not in _model_cache:
        model = YOLO(model_path)
        _model_cache[model_path] = model
        logger.info("YOLO model loaded: %s, device=%s", model_path, device)
    return _model_cache[model_path]


# ---------------------------------------------------------------------------
# Frame sampling
# ---------------------------------------------------------------------------


def _sample_frames(
    video_path: str,
    sample_fps: float,
    duration: float,
) -> tuple[list[tuple[float, Any]], tuple[int, int]]:
    """Sample frames from video at given FPS using OpenCV.

    Returns (frames, frame_shape) where frames is list of (timestamp, frame_array)
    and frame_shape is (height, width).
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if video_fps <= 0:
        video_fps = 30.0
    if duration <= 0:
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / video_fps

    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280

    frame_interval = max(1, int(round(video_fps / sample_fps)))
    frames: list[tuple[float, Any]] = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / video_fps
            frames.append((timestamp, frame))
        frame_idx += 1

    cap.release()
    logger.info(
        "Sampled %d frames from %s (sample_fps=%.1f, duration=%.1fs, shape=%dx%d)",
        len(frames), video_path, sample_fps, duration, frame_w, frame_h,
    )
    return frames, (frame_h, frame_w)


# ---------------------------------------------------------------------------
# YOLO inference (single pass, results mapped to categories)
# ---------------------------------------------------------------------------

# COCO class ID → name mapping (subset used by this pipeline)
COCO_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
    5: "bus", 7: "truck", 16: "dog", 17: "cat",
}

DEFAULT_CATEGORY_CLASSES = {
    "弱势交通参与者异常": [0, 1, 3, 16, 17],      # person, bicycle, motorcycle, dog, cat
    "车辆轨迹与空间冲突": [2, 5, 7, 3],           # car, bus, truck, motorcycle
    "极端高危与失控事件": [0, 1, 2, 3, 5, 7],     # person, bicycle, car, motorcycle, bus, truck
}


def _run_yolo_inference(
    model,
    frames: list[tuple[float, Any]],
    conf_threshold: float,
    iou_threshold: float,
    imgsz: int,
    device: str,
    half: bool,
    batch_size: int,
    category_classes: dict[str, list[int]],
) -> list[dict]:
    """Run YOLO inference on sampled frames (single pass, all categories).

    Returns per-frame results with category-mapped detections.
    """
    results_list: list[dict] = []

    all_images = [f[1] for f in frames]
    all_timestamps = [f[0] for f in frames]

    predict_kwargs = {
        "conf": conf_threshold,
        "iou": iou_threshold,
        "imgsz": imgsz,
        "device": device,
        "half": half,
        "verbose": False,
    }

    if batch_size > 0 and len(all_images) > 1:
        for start in range(0, len(all_images), batch_size):
            end = min(start + batch_size, len(all_images))
            batch_imgs = all_images[start:end]
            batch_ts = all_timestamps[start:end]
            batch_results = model.predict(batch_imgs, **predict_kwargs)
            for ts, result in zip(batch_ts, batch_results):
                frame_data = _parse_yolo_result(ts, result, category_classes)
                results_list.append(frame_data)
    else:
        yolo_results = model.predict(all_images, **predict_kwargs)
        for ts, result in zip(all_timestamps, yolo_results):
            frame_data = _parse_yolo_result(ts, result, category_classes)
            results_list.append(frame_data)

    return results_list


def _parse_yolo_result(
    timestamp: float,
    result: Any,
    category_classes: dict[str, list[int]],
) -> dict:
    """Parse a single YOLO result into structured detection data."""
    detections: list[dict] = []
    boxes = result.boxes
    if boxes is not None and len(boxes) > 0:
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            conf = float(boxes.conf[i])
            xyxy = boxes.xyxy[i].cpu().tolist()  # [x1, y1, x2, y2]
            detections.append({
                "class_id": cls_id,
                "class": COCO_NAMES.get(cls_id, f"class_{cls_id}"),
                "confidence": conf,
                "bbox": xyxy,
            })

    # Map to categories based on class IDs
    category_detections: dict[str, list[dict]] = {}
    for category, class_ids in category_classes.items():
        class_id_set = set(class_ids)
        category_detections[category] = [
            d for d in detections if d["class_id"] in class_id_set
        ]

    return {
        "timestamp": timestamp,
        "detections": detections,
        "category_detections": category_detections,
    }


# ---------------------------------------------------------------------------
# Time window extraction
# ---------------------------------------------------------------------------


def _flag_frames_for_category(
    frame_results: list[dict],
    category: str,
    spatial_rules: dict,
) -> list[tuple[float, str]]:
    """Flag suspicious frames for a given category.

    Returns list of (timestamp, evidence_text).
    """
    flagged: list[tuple[float, str]] = []
    rules = spatial_rules.get(category, {})
    min_objects = int(rules.get("min_objects", 1))
    detection_spike_ratio = float(rules.get("detection_spike_ratio", 0.0))

    # Compute baseline detection count for spike detection
    counts = []
    for fr in frame_results:
        cat_dets = fr.get("category_detections", {}).get(category, [])
        counts.append(len(cat_dets))
    avg_count = sum(counts) / max(1, len(counts)) if counts else 0

    for fr in frame_results:
        ts = fr["timestamp"]
        cat_dets = fr.get("category_detections", {}).get(category, [])
        if not cat_dets:
            continue

        evidence_parts: list[str] = []
        is_flagged = False

        if category == "弱势交通参与者异常":
            # Flag any vulnerable road user detection
            for det in cat_dets:
                is_flagged = True
                evidence_parts.append(f"{det['class']}(conf={det['confidence']:.2f})")

        elif category == "车辆轨迹与空间冲突":
            # Flag if at least min_objects vehicles detected
            if len(cat_dets) >= min_objects:
                is_flagged = True
                class_counts: dict[str, int] = {}
                for det in cat_dets:
                    name = det["class"]
                    class_counts[name] = class_counts.get(name, 0) + 1
                evidence_parts.append(
                    f"{len(cat_dets)} vehicles: " +
                    ", ".join(f"{k}x{v}" for k, v in class_counts.items())
                )

        elif category == "极端高危与失控事件":
            # Flag if any detection exists (at least 1 object)
            if len(cat_dets) >= 1:
                is_flagged = True
                class_counts: dict[str, int] = {}
                for det in cat_dets:
                    name = det["class"]
                    class_counts[name] = class_counts.get(name, 0) + 1
                evidence_parts.append(
                    ", ".join(f"{k}x{v}" for k, v in class_counts.items())
                )
            # Also flag detection count spikes
            count = len(cat_dets)
            if detection_spike_ratio > 0 and avg_count > 0 and count >= avg_count * detection_spike_ratio:
                is_flagged = True
                evidence_parts.append(f"detection spike: {count} objects (avg={avg_count:.1f})")
            # Also flag low-confidence detections (potential partial occlusion / anomaly)
            for det in cat_dets:
                if det["confidence"] < 0.5 and det["class_id"] in (0, 1, 2, 3):
                    is_flagged = True
                    evidence_parts.append(f"low-conf {det['class']}({det['confidence']:.2f})")

        if is_flagged and evidence_parts:
            flagged.append((ts, "；".join(evidence_parts)))

    return flagged


def _merge_flagged_frames(
    flagged: list[tuple[float, str]],
    frame_merge_gap: float,
    window_padding: float,
    min_detection_frames: int,
    duration: float,
) -> list[dict]:
    """Merge flagged frames into time windows."""
    if not flagged:
        return []

    # Sort by timestamp
    flagged.sort(key=lambda x: x[0])

    # Group consecutive flagged frames within frame_merge_gap
    groups: list[list[tuple[float, str]]] = [[flagged[0]]]
    for item in flagged[1:]:
        if item[0] - groups[-1][-1][0] <= frame_merge_gap:
            groups[-1].append(item)
        else:
            groups.append([item])

    # Build windows from groups
    windows: list[dict] = []
    for group in groups:
        if len(group) < min_detection_frames:
            continue
        start = max(0.0, group[0][0] - window_padding)
        end = min(duration, group[-1][0] + window_padding)
        evidence_items = [e for _, e in group]
        unique_evidence = list(dict.fromkeys(evidence_items))
        evidence_text = "；".join(unique_evidence[:5])
        windows.append({
            "start": round(start, 2),
            "end": round(end, 2),
            "reason": f"YOLO: {evidence_text}",
            "suspected_evidence": evidence_text,
        })

    return windows


def _extract_time_windows(
    frame_results: list[dict],
    category_classes: dict[str, list[int]],
    spatial_rules: dict,
    duration: float,
    config: dict,
) -> dict[str, list[dict]]:
    """Extract time windows for each complex category from YOLO results."""
    from nodes.supervisor import _merge_slices_by_iou, _centered_window

    yolo_cfg = config.get("yolo", {})
    frame_merge_gap = float(yolo_cfg.get("frame_merge_gap", 2.0))
    min_detection_frames = int(yolo_cfg.get("min_detection_frames", 2))
    window_padding = float(yolo_cfg.get("window_padding", 1.0))
    max_windows = int(yolo_cfg.get("max_windows_per_category", 3))
    merge_iou = float(
        config.get("tool_call", {})
        .get("fixed_sampling", {})
        .get("profiles", {})
        .get("complex", {})
        .get("first_round", {})
        .get("slice_merge_iou_threshold", 0.4)
    )
    window_seconds = float(
        config.get("tool_call", {})
        .get("fixed_sampling", {})
        .get("profiles", {})
        .get("complex", {})
        .get("first_round", {})
        .get("window_seconds", 8.0)
    )

    result: dict[str, list[dict]] = {}
    for category in category_classes:
        flagged = _flag_frames_for_category(
            frame_results, category, spatial_rules,
        )
        windows = _merge_flagged_frames(
            flagged, frame_merge_gap, window_padding, min_detection_frames, duration,
        )
        if not windows:
            result[category] = []
            continue

        # Apply IoU merge
        windows = _merge_slices_by_iou(windows, duration, window_seconds, merge_iou)

        # Clamp each window to window_seconds
        clamped: list[dict] = []
        for w in windows:
            try:
                center = (float(w["start"]) + float(w["end"])) / 2.0
                left, right = _centered_window(
                    duration=duration,
                    center=center,
                    window=window_seconds,
                )
                clamped.append({
                    "start": round(left, 2),
                    "end": round(right, 2),
                    "reason": w.get("reason", ""),
                    "suspected_evidence": w.get("suspected_evidence", ""),
                })
            except (TypeError, ValueError):
                clamped.append(w)

        result[category] = clamped[:max_windows]

    return result


# ---------------------------------------------------------------------------
# Save detection frames for debugging
# ---------------------------------------------------------------------------


def _save_detection_visualization(
    frames: list[tuple[float, Any]],
    frame_results: list[dict],
    output_dir: str,
) -> None:
    """Save annotated frames with detection boxes for debugging."""
    import cv2

    det_dir = Path(output_dir) / "yolo_detections"
    det_dir.mkdir(parents=True, exist_ok=True)

    for (ts, frame), fr in zip(frames, frame_results):
        dets = fr.get("detections", [])
        if not dets:
            continue
        annotated = frame.copy()
        for det in dets:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            label = f"{det['class']} {det['confidence']:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            )
        fname = f"frame_{ts:.2f}s.jpg"
        cv2.imwrite(str(det_dir / fname), annotated)

    logger.info("Saved annotated frames to %s", det_dir)


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------


def _run_yolo_pipeline(
    video_path: str,
    video_info: dict,
    config: dict,
    output_dir: str,
) -> tuple[dict, dict]:
    """Synchronous YOLO pipeline (frame sampling + inference + window extraction).

    Returns (yolo_detections, yolo_raw_results).
    """
    yolo_cfg = config.get("yolo", {})
    model_path = str(yolo_cfg.get("model_path", "/mnt/data/checkpoints/yolo/yolo11s.pt"))
    device = str(yolo_cfg.get("device", "0"))
    if device == "auto":
        try:
            import torch
            device = "0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    sample_fps = float(yolo_cfg.get("sample_fps", 2.0))
    conf_threshold = float(yolo_cfg.get("conf_threshold", 0.35))
    iou_threshold = float(yolo_cfg.get("iou_threshold", 0.45))
    imgsz = int(yolo_cfg.get("imgsz", 640))
    batch_size = int(yolo_cfg.get("batch_size", 8))
    half = bool(yolo_cfg.get("half", True)) and device != "cpu"
    save_frames = bool(yolo_cfg.get("save_detection_frames", False))

    category_classes = yolo_cfg.get("category_classes", DEFAULT_CATEGORY_CLASSES)
    spatial_rules = yolo_cfg.get("category_spatial_rules", {})

    duration = float(video_info.get("duration", 0) or 0)

    # Load model (try primary path, fallback to secondary)
    model_path_fallback = str(yolo_cfg.get("model_path_fallback", ""))
    model = None
    used_path = model_path
    if Path(model_path).exists():
        model = _load_yolo_model(model_path, device)
    elif model_path_fallback and Path(model_path_fallback).exists():
        logger.warning("Primary model not found: %s, using fallback: %s", model_path, model_path_fallback)
        model = _load_yolo_model(model_path_fallback, device)
        used_path = model_path_fallback
    else:
        logger.error("No YOLO model found: tried %s%s", model_path,
                     f", {model_path_fallback}" if model_path_fallback else "")
        return {}, {}

    # Sample frames
    frames, frame_shape = _sample_frames(video_path, sample_fps, duration)
    if not frames:
        logger.warning("No frames sampled from %s", video_path)
        return {}, {}

    # Run YOLO inference (single pass for all categories)
    frame_results = _run_yolo_inference(
        model, frames, conf_threshold, iou_threshold, imgsz, device, half,
        batch_size, category_classes,
    )

    # Save debug visualization
    if save_frames:
        _save_detection_visualization(frames, frame_results, output_dir)

    # Extract time windows per category
    yolo_detections = _extract_time_windows(
        frame_results, category_classes, spatial_rules, duration, config,
    )

    # Raw results for debugging (lightweight summary)
    yolo_raw_results = {
        "model": used_path,
        "device": device,
        "sample_fps": sample_fps,
        "frame_shape": list(frame_shape),
        "total_frames_sampled": len(frames),
        "frame_results": [
            {
                "timestamp": fr["timestamp"],
                "total_detections": len(fr["detections"]),
                "category_counts": {
                    cat: len(dets) for cat, dets in fr.get("category_detections", {}).items()
                },
            }
            for fr in frame_results
        ],
        "windows": yolo_detections,
    }

    logger.info(
        "YOLO pre-filter complete: %s",
        {cat: len(wins) for cat, wins in yolo_detections.items()},
    )

    return yolo_detections, yolo_raw_results


async def yolo_pre_filter_node(state: dict) -> dict:
    """YOLO pre-filter node for the LangGraph pipeline.

    Runs in parallel with simple_workers. Samples video frames, runs single-pass
    YOLO detection, and extracts suspicious time windows for complex categories.

    When YOLO finds no detections for a category, supervisor_2 will output
    a direct "normal" result for that category without invoking LLM.

    Input state keys:
    - video_path: str
    - video_info: dict (duration, fps)
    - config: dict
    - output_dir: str

    Output state keys:
    - yolo_detections: dict (category -> list[time_slice])
    - yolo_raw_results: dict (raw per-frame detection summary)
    """
    config = state.get("config", {})
    yolo_cfg = config.get("yolo", {})
    enabled = bool(yolo_cfg.get("enabled", False))

    if not enabled:
        logger.info("YOLO pre-filter disabled, skipping")
        return {"yolo_detections": {}, "yolo_raw_results": {}}

    video_path = state.get("video_path", "")
    video_info = state.get("video_info", {})
    output_dir = state.get("output_dir", "")

    # Build full video path
    video_root = config.get("paths", {}).get("video_root", "")
    full_video_path = str(Path(video_root) / video_path)

    if not Path(full_video_path).exists():
        logger.warning("Video file not found: %s, YOLO skipped", full_video_path)
        return {"yolo_detections": {}, "yolo_raw_results": {}}

    try:
        # Run YOLO pipeline in thread pool to avoid blocking event loop
        yolo_detections, yolo_raw_results = await asyncio.to_thread(
            _run_yolo_pipeline,
            full_video_path,
            video_info,
            config,
            output_dir,
        )
        return {
            "yolo_detections": yolo_detections,
            "yolo_raw_results": yolo_raw_results,
        }
    except Exception as e:
        logger.error("YOLO pre-filter error: %s", e, exc_info=True)
        fallback = bool(yolo_cfg.get("fallback_to_code", True))
        if fallback:
            logger.info("YOLO failed, will fallback to code mode in supervisor_2")
        return {"yolo_detections": {}, "yolo_raw_results": {}}
