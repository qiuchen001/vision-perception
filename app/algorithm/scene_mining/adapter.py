import asyncio
import hashlib
import json
import os
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import requests
import yaml

from config.config import Config


SCENE_MINING_DIR = Path(__file__).resolve().parent
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


@dataclass
class SceneMiningResult:
    video_path: str
    local_video_path: str
    output_dir: str
    final_output: dict[str, Any]
    summary_item: dict[str, Any]
    raw_outputs: dict[str, str]
    config: dict[str, Any]


@contextmanager
def _scene_mining_import_path():
    path = str(SCENE_MINING_DIR)
    inserted = False
    if path not in sys.path:
        sys.path.insert(0, path)
        inserted = True
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(path)
            except ValueError:
                pass


def _resolve_path(path_value: str, config_dir: Path) -> str:
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    copied_path = SCENE_MINING_DIR / path
    if copied_path.exists():
        return str(copied_path.resolve())
    return str((config_dir / path).resolve())


def load_scene_mining_config(
    config_path: str | None = None,
    *,
    video_root: str | None = None,
    video_url_prefix: str | None = None,
    output_base: str | None = None,
) -> dict[str, Any]:
    resolved_config_path = Path(config_path or Config.SCENE_MINING_CONFIG_PATH)
    if not resolved_config_path.exists():
        fallback = SCENE_MINING_DIR / "config-qwen-gemini.yaml"
        if not fallback.exists():
            raise FileNotFoundError(f"场景挖掘配置文件不存在: {resolved_config_path}")
        resolved_config_path = fallback

    with resolved_config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    api_cfg = config.setdefault("api", {})
    if os.getenv("SCENE_MINING_API_BASE_URL"):
        api_cfg["base_url"] = os.getenv("SCENE_MINING_API_BASE_URL")
    if os.getenv("SCENE_MINING_API_MODEL_NAME"):
        api_cfg["model_name"] = os.getenv("SCENE_MINING_API_MODEL_NAME")

    config_dir = resolved_config_path.parent
    paths_cfg = config.setdefault("paths", {})
    paths_cfg["categories"] = _resolve_path(paths_cfg.get("categories", "categories.json"), config_dir)
    paths_cfg["skills"] = _resolve_path(paths_cfg.get("skills", "skills"), config_dir)
    paths_cfg["output_base"] = str(Path(output_base or Config.SCENE_MINING_OUTPUT_DIR).resolve())
    paths_cfg["tool_clip_root"] = str(
        Path(
            os.getenv(
                "SCENE_MINING_TOOL_CLIP_DIR",
                str(Path(paths_cfg["output_base"]) / "_tool_clips"),
            )
        ).resolve()
    )
    if video_root:
        paths_cfg["video_root"] = str(Path(video_root).resolve())
    else:
        paths_cfg["video_root"] = str(Path(paths_cfg["video_root"]).resolve())
    paths_cfg["video_url_prefix"] = (
        video_url_prefix
        or os.getenv("SCENE_MINING_VIDEO_URL_PREFIX")
        or Config.SCENE_MINING_VIDEO_URL_PREFIX
    ).rstrip("/")

    tool_cfg = config.setdefault("tool_call", {})
    tool_cfg["clip_tool_path"] = _resolve_path(tool_cfg.get("clip_tool_path", "select_clip_fallback.py"), config_dir)
    return config


def _download_video(video_url: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    parsed = urlparse(video_url)
    suffix = Path(parsed.path).suffix or ".mp4"
    digest = hashlib.sha256(video_url.encode("utf-8")).hexdigest()[:16]
    target = cache_dir / f"{digest}{suffix}"
    if target.exists() and target.stat().st_size > 0:
        return target

    with requests.get(video_url, stream=True, timeout=60) as response:
        response.raise_for_status()
        temp_target = target.with_suffix(f"{target.suffix}.tmp")
        with temp_target.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        temp_target.replace(target)
    return target


def _path_from_file_url(video_url: str) -> Path:
    parsed = urlparse(video_url)
    return Path(unquote(parsed.path))


def _prepare_video(
    video_path_or_url: str,
    config: dict[str, Any],
) -> tuple[str, str, str, dict[str, Any]]:
    raw_value = str(video_path_or_url or "").strip()
    if not raw_value:
        raise ValueError("视频路径不能为空")

    parsed = urlparse(raw_value)
    if parsed.scheme in {"http", "https"}:
        local_path = _download_video(raw_value, Path(Config.SCENE_MINING_VIDEO_CACHE_DIR))
        config = dict(config)
        config["paths"] = dict(config.get("paths", {}))
        config["paths"]["video_root"] = str(local_path.parent.resolve())
        config["paths"]["video_url_prefix"] = f"file://{local_path.parent.resolve()}"
        return local_path.name, str(local_path), raw_value, config

    local_path = _path_from_file_url(raw_value) if parsed.scheme == "file" else Path(raw_value)
    if not local_path.exists():
        raise FileNotFoundError(f"视频文件不存在或算法端无法访问: {raw_value}")

    root = Path(config["paths"]["video_root"]).resolve()
    local_path = local_path.resolve()
    try:
        relative_video_path = local_path.relative_to(root).as_posix()
    except ValueError:
        config = dict(config)
        config["paths"] = dict(config.get("paths", {}))
        config["paths"]["video_root"] = str(local_path.parent.resolve())
        config["paths"]["video_url_prefix"] = f"file://{local_path.parent.resolve()}"
        relative_video_path = local_path.name
    return relative_video_path, str(local_path), str(local_path), config


def strip_model_thinking(text: str) -> str:
    stripped = _THINK_RE.sub("", text or "")
    stripped = re.sub(r"</?think>", "", stripped, flags=re.IGNORECASE)
    return stripped.strip()


def build_summary_item(final_output: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    pred_results: dict[str, list[str]] = {}
    for category_name, category_result in final_output.items():
        pred = category_result.get("pred", []) if isinstance(category_result, dict) else []
        pred_results[category_name] = pred if isinstance(pred, list) else []

    abnormal_categories = set(config.get("pipeline", {}).get("abnormal_categories", []))
    events: list[dict[str, Any]] = []
    for category_name, category_result in final_output.items():
        if category_name not in abnormal_categories or not isinstance(category_result, dict):
            continue
        for event in category_result.get("events", []) or []:
            if isinstance(event, dict):
                events.append(dict(event))

    return {"pred": pred_results, "abnormal_event_times": events}


def extract_tags(summary_item: dict[str, Any]) -> list[str]:
    pred = summary_item.get("pred", {})
    tags: list[str] = []
    if not isinstance(pred, dict):
        return tags
    for values in pred.values():
        if not isinstance(values, list):
            continue
        for value in values:
            tag = str(value).strip()
            if tag and tag not in tags:
                tags.append(tag)
    return tags


def _read_raw_outputs(video_output_dir: Path) -> dict[str, str]:
    raw_outputs: dict[str, str] = {}
    agent_outputs_file = video_output_dir / "agent_outputs.json"
    if agent_outputs_file.exists():
        data = json.loads(agent_outputs_file.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for item in data.values():
                if not isinstance(item, dict):
                    continue
                category = str(item.get("task_name") or item.get("agent_name") or "").strip()
                if " [" in category:
                    category = category.split(" [", 1)[0].strip()
                raw_content = strip_model_thinking(str(item.get("raw_content") or ""))
                if category and raw_content:
                    raw_outputs.setdefault(category, raw_content)

    for raw_file in video_output_dir.glob("*_raw.txt"):
        category = raw_file.name[:-8]
        raw_content = strip_model_thinking(raw_file.read_text(encoding="utf-8"))
        if raw_content:
            raw_outputs.setdefault(category, raw_content)
    return raw_outputs


async def analyze_video_async(
    video_path_or_url: str,
    *,
    config_path: str | None = None,
    output_base: str | None = None,
    progress_callback=None,
) -> SceneMiningResult:
    config = load_scene_mining_config(config_path, output_base=output_base)
    relative_video_path, local_video_path, _, config = _prepare_video(video_path_or_url, config)
    output_base_path = Path(config["paths"]["output_base"])
    run_output_dir = output_base_path / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_output_dir.mkdir(parents=True, exist_ok=True)

    with _scene_mining_import_path():
        from langgraph_pipeline import run_react_video_analysis, run_video_analysis
        from prompts import (
            build_complex_system_prompt,
            build_global_system_prompt,
            build_simple_system_prompt,
            init_prompts,
        )
        from qwen_client import QwenClient

        init_prompts(config["paths"]["categories"], config["paths"]["skills"])
        client = QwenClient(config)
        client.init_system_prompt(build_global_system_prompt(config["paths"]["skills"]))
        client.init_split_system_prompts(
            simple_prompt=build_simple_system_prompt(config["paths"]["skills"]),
            complex_prompt=build_complex_system_prompt(config["paths"]["skills"]),
        )
        try:
            agent_mode = str(config.get("pipeline", {}).get("agent_mode", "dag")).lower()
            if agent_mode == "react":
                final_output = await run_react_video_analysis(
                    video_path=relative_video_path,
                    video_root=config["paths"]["video_root"],
                    output_dir=str(run_output_dir),
                    qwen_client=client,
                    config=config,
                    skills_dir=config["paths"]["skills"],
                    previous_preds=None,
                    on_step=progress_callback,
                )
            else:
                final_output = await run_video_analysis(
                    video_path=relative_video_path,
                    video_root=config["paths"]["video_root"],
                    output_dir=str(run_output_dir),
                    qwen_client=client,
                    config=config,
                    skills_dir=config["paths"]["skills"],
                    previous_preds=None,
                    on_step=progress_callback,
                )
        finally:
            await client.close()

    summary_item = build_summary_item(final_output, config)
    video_output_dir = run_output_dir / Path(relative_video_path).stem
    raw_outputs = _read_raw_outputs(video_output_dir)
    return SceneMiningResult(
        video_path=relative_video_path,
        local_video_path=local_video_path,
        output_dir=str(video_output_dir),
        final_output=final_output,
        summary_item=summary_item,
        raw_outputs=raw_outputs,
        config=config,
    )


def analyze_video(
    video_path_or_url: str,
    *,
    config_path: str | None = None,
    output_base: str | None = None,
    progress_callback=None,
) -> SceneMiningResult:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            analyze_video_async(
                video_path_or_url,
                config_path=config_path,
                output_base=output_base,
                progress_callback=progress_callback,
            )
        )
    raise RuntimeError("analyze_video 不能在已运行的事件循环中同步调用，请使用 analyze_video_async")
