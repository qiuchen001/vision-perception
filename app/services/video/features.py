import json
import math
import os
import re
import subprocess
import tempfile
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

from app.dao.feature_dao import FeatureDAO
from app.utils.embedding.embedding_factory import EmbeddingFactory
from config.config import Config


class VideoFeatureService:
    def __init__(self):
        self.dao = FeatureDAO()
        self.dao.ensure_collections()
        self.embedding_model = EmbeddingFactory.create_embedding()

    def upsert_text_features(self, video: Dict[str, Any]) -> None:
        tags = video.get("tags") or []
        tags_text = "、".join(str(tag) for tag in tags if str(tag).strip())
        summary_text = video.get("summary_txt") or ""
        if not tags_text or not summary_text:
            raise ValueError("生成文本特征需要 tags 和 summary_txt")
        tags_embedding = self.embedding_model.embedding_text(tags_text)
        summary_embedding = self.embedding_model.embedding_text(summary_text)
        self.dao.upsert_text_features(video, tags_text, summary_text, tags_embedding, summary_embedding)

    def upsert_visual_feature(self, video: Dict[str, Any]) -> None:
        visual = VisualFeatureBuilder(self.embedding_model).build(video)
        self.dao.upsert_visual_feature(
            video=video,
            embedding=visual["embedding"],
            sampling_policy=visual["sampling_policy"],
            sampled_seconds=json.dumps(visual["sampled_seconds"], ensure_ascii=False),
            sampled_frame_count=len(visual["sampled_seconds"]),
        )


class VisualFeatureBuilder:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def build(self, video: Dict[str, Any]) -> Dict[str, Any]:
        video_path = self._select_video_path(video)
        duration = self._probe_duration(video_path)
        event_windows = self._extract_event_windows(video.get("mining_results"))
        samples = self._build_sample_plan(duration, event_windows)
        if not samples:
            raise ValueError(f"无法为视频生成采样点: {video_path}")

        embeddings: List[np.ndarray] = []
        weights: List[float] = []
        sampled_seconds: List[float] = []
        with tempfile.TemporaryDirectory(prefix="vision_perception_frames_") as tmpdir:
            for idx, (second, weight) in enumerate(samples):
                frame_path = os.path.join(tmpdir, f"frame_{idx:04d}.jpg")
                if not self._extract_frame(video_path, second, frame_path):
                    continue
                with Image.open(frame_path) as image:
                    emb = np.asarray(self.embedding_model.embedding_image(image.convert("RGB")), dtype=np.float32)
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                embeddings.append(emb)
                weights.append(weight)
                sampled_seconds.append(round(float(second), 3))

        if not embeddings:
            raise ValueError(f"视频抽帧全部失败，无法生成视觉特征: {video_path}")

        matrix = np.vstack(embeddings)
        weight_arr = np.asarray(weights, dtype=np.float32)
        pooled = np.average(matrix, axis=0, weights=weight_arr)
        pooled_norm = np.linalg.norm(pooled)
        if pooled_norm > 0:
            pooled = pooled / pooled_norm

        return {
            "embedding": pooled.astype(float).tolist(),
            "sampled_seconds": sampled_seconds,
            "sampling_policy": (
                f"ffmpeg_uniform_fps={Config.FRAME_SAMPLE_FPS},max_frames={Config.FRAME_SAMPLE_MAX_FRAMES},"
                f"event_fps={Config.FRAME_SAMPLE_EVENT_FPS},event_weight={Config.FRAME_SAMPLE_EVENT_WEIGHT}"
            ),
        }

    @staticmethod
    def _select_video_path(video: Dict[str, Any]) -> str:
        mining_results = video.get("mining_results")
        if isinstance(mining_results, dict):
            for key in ("local_video_path", "video_path"):
                value = mining_results.get(key)
                if value and os.path.exists(str(value)):
                    return str(value)
        path = str(video.get("path") or "")
        if path.startswith("file://"):
            return path[7:]
        return path

    @staticmethod
    def _probe_duration(video_path: str) -> float:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration = float(result.stdout.strip())
        if duration <= 0:
            raise ValueError(f"视频 duration 非法: {duration}")
        return duration

    @staticmethod
    def _extract_frame(video_path: str, second: float, frame_path: str) -> bool:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{second:.3f}",
            "-i",
            video_path,
            "-frames:v",
            "1",
            "-q:v",
            "2",
            frame_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0 and os.path.exists(frame_path) and os.path.getsize(frame_path) > 0

    @staticmethod
    def _build_sample_plan(duration: float, event_windows: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        max_frames = max(1, Config.FRAME_SAMPLE_MAX_FRAMES)
        uniform_interval = max(1.0 / max(Config.FRAME_SAMPLE_FPS, 0.001), duration / max_frames)
        samples: Dict[float, float] = {}

        t = 0.0
        while t < duration and len(samples) < max_frames:
            samples[round(t, 3)] = 1.0
            t += uniform_interval
        samples[round(max(0.0, min(duration - 0.01, duration)), 3)] = 1.0

        event_interval = 1.0 / max(Config.FRAME_SAMPLE_EVENT_FPS, 0.001)
        for start, end in event_windows:
            left = max(0.0, start - 1.0)
            right = min(duration, max(end, start) + 1.0)
            t = left
            while t <= right:
                key = round(max(0.0, min(duration - 0.01, t)), 3)
                samples[key] = max(samples.get(key, 1.0), Config.FRAME_SAMPLE_EVENT_WEIGHT)
                t += event_interval

        ordered = sorted(samples.items(), key=lambda item: item[0])
        if len(ordered) > max_frames:
            step = len(ordered) / max_frames
            picked = [ordered[min(math.floor(i * step), len(ordered) - 1)] for i in range(max_frames)]
            ordered = sorted(dict(picked).items(), key=lambda item: item[0])
        return ordered

    @classmethod
    def _extract_event_windows(cls, mining_results: Any) -> List[Tuple[float, float]]:
        if not isinstance(mining_results, dict):
            return []
        events = mining_results.get("abnormal_event_times") or []
        windows = []
        for event in events:
            if isinstance(event, dict):
                parsed = cls._event_dict_to_window(event)
                if parsed:
                    windows.append(parsed)
            elif isinstance(event, str):
                parsed = cls._event_text_to_window(event)
                if parsed:
                    windows.append(parsed)
        return windows

    @staticmethod
    def _event_dict_to_window(event: Dict[str, Any]) -> Tuple[float, float] | None:
        start_keys = ("start", "start_time", "start_sec", "start_second", "begin", "begin_time")
        end_keys = ("end", "end_time", "end_sec", "end_second", "finish", "finish_time")
        start = next((event.get(k) for k in start_keys if event.get(k) is not None), None)
        end = next((event.get(k) for k in end_keys if event.get(k) is not None), None)
        if start is None and end is None:
            return VisualFeatureBuilder._event_text_to_window(json.dumps(event, ensure_ascii=False))
        try:
            start_f = float(start if start is not None else end)
            end_f = float(end if end is not None else start)
            return min(start_f, end_f), max(start_f, end_f)
        except (TypeError, ValueError):
            return VisualFeatureBuilder._event_text_to_window(json.dumps(event, ensure_ascii=False))

    @staticmethod
    def _event_text_to_window(text: str) -> Tuple[float, float] | None:
        numbers = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*s", text)]
        if not numbers:
            numbers = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)", text)]
        if not numbers:
            return None
        if len(numbers) == 1:
            return numbers[0], numbers[0]
        return min(numbers[0], numbers[1]), max(numbers[0], numbers[1])
