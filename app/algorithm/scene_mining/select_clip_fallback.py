import argparse
import datetime
import json
import subprocess
import uuid
from pathlib import Path
from typing import Optional


def _parse_time_to_seconds(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None
    if ":" not in value:
        return float(value)
    parts = value.split(":")
    total = 0.0
    for i, part in enumerate(reversed(parts)):
        total += float(part) * (60**i)
    return total


def _probe_duration(video_path: str) -> float:
    proc = subprocess.run(
        [
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
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(proc.stdout or "{}")
    streams = data.get("streams", [])
    if not streams:
        return 0.0
    return float(streams[0].get("duration", 0.0) or 0.0)


def _normalize_range(start_time: float, end_time: Optional[float], duration: float, clamp: bool) -> tuple[float, float]:
    actual_end = duration if end_time is None else end_time
    if duration > 0 and clamp:
        start_time = max(0.0, min(start_time, duration))
        actual_end = max(start_time + 0.1, min(actual_end, duration))
    if actual_end <= start_time:
        raise ValueError(f"Invalid time range: start={start_time}, end={actual_end}")
    return start_time, actual_end


def _build_out_path(video_path: str, save_root: str, save_dir: Optional[str]) -> Path:
    if save_dir:
        out_dir = Path(save_dir)
    else:
        date_str = datetime.datetime.now().strftime("%Y%m%d")
        time_str = datetime.datetime.now().strftime("%H%M%S")
        video_name = Path(video_path).stem
        rand = str(uuid.uuid4())[:8]
        out_dir = Path(save_root) / date_str / video_name / f"{time_str}_{rand}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "clip.mp4"


def extract_clip(video_path: str, start_time: float, end_time: float, out_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_time:.3f}",
        "-to",
        f"{end_time:.3f}",
        "-i",
        video_path,
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "ffmpeg failed")
    if not out_path.exists() or out_path.stat().st_size < 1024:
        raise RuntimeError(f"clip文件无效或过小: {out_path} ({out_path.stat().st_size if out_path.exists() else '不存在'}字节)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract video clip fallback tool")
    parser.add_argument("--video-path", "--video_path", required=True)
    parser.add_argument("--start-time", "--start_time", type=str, default="0")
    parser.add_argument("--end-time", "--end_time", type=str, default=None)
    parser.add_argument("--sampling-interval", "--sampling_interval", type=float, default=0.2)
    parser.add_argument("--save-root", default="/tmp")
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--clamp", action="store_true", default=False)
    args = parser.parse_args()

    start_time = _parse_time_to_seconds(args.start_time)
    end_time = _parse_time_to_seconds(args.end_time)
    if start_time is None:
        raise ValueError("start_time is required")
    duration = _probe_duration(args.video_path)
    start_time, end_time = _normalize_range(start_time, end_time, duration, args.clamp)
    out_path = _build_out_path(args.video_path, args.save_root, args.save_dir)
    extract_clip(args.video_path, start_time, end_time, out_path)
    print(
        f"Clip saved to {out_path}, start={start_time:.3f}, end={end_time:.3f}, sampling_interval={args.sampling_interval:.3f}"
    )


if __name__ == "__main__":
    main()
