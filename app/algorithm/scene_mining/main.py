import argparse
import asyncio
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml
from tqdm import tqdm

import prompts
from prompts import get_categories, init_prompts, build_global_system_prompt, build_simple_system_prompt, build_complex_system_prompt, build_react_system_prompt, is_abnormal_category, get_sorted_categories, load_category_skill, get_skills_dir
from qwen_client import QwenClient
from response_parser import ResponseParser
from langgraph_pipeline import run_video_analysis, run_react_video_analysis, get_video_info as _get_video_info_from_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Interactive Printer (Claude Code style output)
# ---------------------------------------------------------------------------

class InteractivePrinter:
    """Print ReAct agent steps to terminal in a readable format."""

    def __init__(self, show_thinking: bool = True):
        self.show_thinking = show_thinking
        self._use_rich = False
        try:
            from rich.console import Console
            from rich.panel import Panel
            self._console = Console()
            self._Panel = Panel
            self._use_rich = True
        except ImportError:
            pass

    def __call__(self, step_type: str, data: dict):
        if step_type == "start":
            self._print_panel("开始", data.get("message", ""), style="bold green")
        elif step_type == "thinking":
            if self.show_thinking:
                content = data.get("content", "")
                if content:
                    self._print_panel("思考", content[:500], style="cyan")
        elif step_type == "tool_call":
            tool = data.get("tool", "?")
            args = data.get("args", {})
            args_str = json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else str(args)
            self._print_panel("工具调用", f"{tool}({args_str})", style="yellow")
        elif step_type == "tool_result":
            tool = data.get("tool", "")
            output = data.get("output", "")
            prefix = f"[{tool}] " if tool else ""
            self._print_panel("工具结果", f"{prefix}{output[:300]}", style="green")
        elif step_type == "final":
            result = data.get("result", {})
            pred = result.get("pred", [])
            self._print_panel("结果", f"pred: {json.dumps(pred, ensure_ascii=False)}", style="bold magenta")
        elif step_type == "complete":
            msg = data.get("message", "")
            results = data.get("results", {})
            lines = [msg]
            for cat, pred in results.items():
                lines.append(f"  {cat}: {json.dumps(pred, ensure_ascii=False)}")
            self._print_panel("分析完成", "\n".join(lines), style="bold green")

    def _print_panel(self, title: str, content: str, style: str = ""):
        if self._use_rich:
            from rich.panel import Panel
            panel = Panel(content, title=title, border_style=style, expand=False)
            self._console.print(panel)
        else:
            # Simple terminal output
            width = 60
            print(f"\n╭─ {title} " + "─" * max(0, width - len(title) - 3) + "╮")
            for line in content.split("\n"):
                print(f"│ {line}")
            print("╰" + "─" * (width + 1) + "╯")


def configure_runtime_logging(is_test_mode: bool) -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO if is_test_mode else logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.ERROR)
    logging.getLogger("httpcore").setLevel(logging.ERROR)
    logging.getLogger("openai").setLevel(logging.ERROR)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_git_commit() -> str:
    """Return '<short-hash>[-dirty]' of the current HEAD, or 'unknown' on failure."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty = subprocess.call(
            ["git", "diff", "--quiet", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return f"{sha}-dirty" if dirty else sha
    except Exception:
        return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--use-langgraph", action="store_true", help="Use new LangGraph multi-agent pipeline")
    parser.add_argument("--legacy-pipeline", action="store_true", help="Force original non-LangGraph pipeline")
    parser.add_argument("--agent-mode", choices=["dag", "react"], default=None,
                        help="Agent mode: dag=original pipeline, react=new ReAct pipeline")
    parser.add_argument("--show-thinking", action="store_true",
                        help="Show model's thinking process in real-time (single video mode)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from an existing output directory (skips already completed videos, reuses its config)")
    return parser.parse_args()


def _apply_overrides(config: dict, set_args: list[str]) -> dict:
    for item in set_args:
        if "=" not in item:
            raise ValueError(f"无效 --set 参数（缺少 '='）: {item}")
        dotted_key, raw_value = item.split("=", 1)
        keys = [k for k in dotted_key.strip().split(".") if k]
        if not keys:
            raise ValueError(f"无效 --set 参数（空 key）: {item}")
        value = yaml.safe_load(raw_value)
        node = config
        for key in keys[:-1]:
            if key not in node or not isinstance(node[key], dict):
                node[key] = {}
            node = node[key]
        node[keys[-1]] = value
    return config


def save_config(config: dict, output_dir: Path) -> None:
    config_file = output_dir / "config.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)


def scan_completed_videos(output_dir: Path, annotations: list[dict], video_root: str) -> tuple[list[dict], dict]:
    """Scan output_dir for already-completed videos and return remaining annotations + existing summary.

    A video is considered complete if its entry exists in summary.json AND
    the per-video result.json is a valid JSON file.

    Returns:
        remaining: annotations not yet completed
        existing_summary: loaded summary dict from previous run
    """
    summary_file = output_dir / "summary.json"
    existing_summary: dict = {}
    if summary_file.exists():
        try:
            existing_summary = json.loads(summary_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("无法解析已有 summary.json，将从头开始: %s", e)
            existing_summary = {}

    completed_stems: set[str] = set()
    for ann in annotations:
        video_path = ann["video_path"]
        stem = Path(video_path).stem
        output_key = build_output_key(video_root, video_path)
        # Must exist in summary AND have a valid per-video result
        if output_key not in existing_summary:
            continue
        result_file = output_dir / stem / "result.json"
        if not result_file.exists():
            continue
        try:
            json.loads(result_file.read_text(encoding="utf-8"))
            completed_stems.add(video_path)
        except (json.JSONDecodeError, OSError):
            logger.warning("result.json 损坏，将重新处理: %s", stem)

    remaining = [a for a in annotations if a["video_path"] not in completed_stems]
    logger.info("断点续传: 已完成 %d / %d，剩余 %d 待处理",
                len(annotations) - len(remaining), len(annotations), len(remaining))
    return remaining, existing_summary


def load_annotations(annotations_path: str) -> list[dict]:
    annotations = []
    with open(annotations_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                annotations.append(json.loads(line))
    return annotations


def _normalize_video_path(video_path: str, video_root: str) -> str | None:
    raw_path = video_path.strip()
    if not raw_path:
        return None

    root = Path(video_root).resolve()
    path_obj = Path(raw_path)

    if path_obj.is_absolute():
        try:
            return path_obj.resolve().relative_to(root).as_posix()
        except ValueError:
            logger.warning("跳过不在 video_root 下的视频路径: %s", raw_path)
            return None
    return path_obj.as_posix()


def load_video_paths(video_paths_file: str, video_root: str) -> list[dict]:
    annotations: list[dict] = []
    seen: set[str] = set()
    with open(video_paths_file, "r", encoding="utf-8") as f:
        for line in f:
            normalized = _normalize_video_path(line, video_root)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            annotations.append({"video_path": normalized})
    return annotations


def build_video_url(video_root: str, video_path: str, config: dict | None = None) -> str:
    prefix = "file:///app/videos"
    if config:
        prefix = str(config.get("paths", {}).get("video_url_prefix", prefix))
    return f"{prefix.rstrip('/')}/{video_path}"


def build_output_key(video_root: str, video_path: str) -> str:
    return str(Path(video_root) / video_path)


def _save_per_video_result(output_dir: Path, output_key: str, results: dict) -> None:
    """Write result.json for a single video (all pipeline modes)."""
    stem = Path(output_key).stem
    video_output_dir = output_dir / stem
    video_output_dir.mkdir(parents=True, exist_ok=True)
    result_file = video_output_dir / "result.json"
    write_json_atomic(result_file, results)


def write_json_atomic(file_path: Path, data: dict) -> None:
    """Write JSON atomically using a temp file + rename to prevent partial writes."""
    temp_file = file_path.with_suffix(f"{file_path.suffix}.tmp")
    temp_file.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_file.replace(file_path)


def _get_video_duration(video_path: str) -> float:
    """Return video duration in seconds via the shared pipeline utility."""
    return _get_video_info_from_pipeline(video_path).get("duration", 0.0)


def _normalize_event_time_range(start_val: float, end_val: float, duration: float) -> tuple[float, float]:
    start = float(start_val)
    end = float(end_val)
    if end < start:
        start, end = end, start
    if duration <= 0:
        return max(0.0, start), max(0.0, end)
    span = max(0.1, end - start)
    if end <= duration and start >= 0:
        return start, end
    if start >= duration:
        end = duration
        start = max(0.0, duration - span)
        return start, max(start + 0.1, end)
    start = max(0.0, start)
    end = min(duration, end)
    if end <= start:
        end = min(duration, start + 0.1)
    return start, end


def _merge_abnormal_events(events: list[dict], gap_merge_threshold: float = 1.5) -> list[dict]:
    """Merge same-type events across all categories by time proximity.

    Events of the same type that overlap or are within gap_merge_threshold seconds
    are merged into a single event spanning their union.
    Different-type events are always kept separate.
    """
    if not events:
        return events

    type_groups: dict[str, list[dict]] = {}
    for evt in events:
        evt_type = str(evt.get("type") or "")
        type_groups.setdefault(evt_type, []).append(evt)

    result: list[dict] = []
    for evt_type, group in type_groups.items():
        group.sort(key=lambda e: float(e.get("start_time", 0.0)))
        merged = [dict(group[0])]
        for evt in group[1:]:
            last = merged[-1]
            gap = float(evt.get("start_time", 0.0)) - float(last["end_time"])
            if gap <= gap_merge_threshold:
                last["start_time"] = min(float(last["start_time"]), float(evt.get("start_time", 0.0)))
                last["end_time"] = max(float(last["end_time"]), float(evt.get("end_time", 0.0)))
            else:
                merged.append(dict(evt))
        result.extend(merged)

    result.sort(key=lambda e: float(e.get("start_time", 0.0)))
    return result


def build_summary_item(results: dict, config: dict, video_duration: float = 0.0) -> dict:
    pred_results = {}
    for category_name, category_result in results.items():
        if isinstance(category_result, dict):
            pred = category_result.get("pred", [])
            pred_results[category_name] = pred if isinstance(pred, list) else []

    raw_events = []
    for category_name, category_result in results.items():
        if is_abnormal_category(category_name, config):
            if isinstance(category_result, dict):
                events = category_result.get("events", [])
                if isinstance(events, list):
                    for event in events:
                        if not isinstance(event, dict):
                            continue
                        try:
                            start_val = float(event.get("start_time", 0.0))
                            end_val = float(event.get("end_time", start_val))
                        except (TypeError, ValueError):
                            continue
                        start_val, end_val = _normalize_event_time_range(start_val, end_val, video_duration)
                        raw_events.append(
                            {
                                "type": event.get("type"),
                                "start_time": start_val,
                                "end_time": end_val,
                            }
                        )

    abnormal_event_times = _merge_abnormal_events(raw_events)

    return {
        "pred": pred_results,
        "abnormal_event_times": abnormal_event_times,
    }


async def process_video(
    annotation: dict,
    output_dir: Path,
    client: QwenClient,
    config: dict,
    detailed: bool = False,
) -> tuple[str, dict]:
    video_root = config["paths"]["video_root"]
    video_path = annotation["video_path"]
    video_url = build_video_url(video_root, video_path, config)
    output_key = build_output_key(video_root, video_path)

    video_output_dir = output_dir / Path(video_path).stem
    video_output_dir.mkdir(parents=True, exist_ok=True)

    # Fetch duration once per video so per-category calls have the right value.
    video_duration = _get_video_duration(output_key)

    categories = get_categories()
    category_order = get_sorted_categories(config)
    skills_dir = get_skills_dir()

    parsed_results: dict[str, dict] = {}
    previous_preds: dict[str, list] = {}
    pipeline_cfg = config.get("pipeline", {})
    category_execution_mode = str(pipeline_cfg.get("category_execution_mode", "parallel")).lower()
    category_concurrency = int(pipeline_cfg.get("category_concurrency", max(1, len(category_order))))

    async def _run_one_category(category_name: str, inherited_preds: dict[str, list]):
        is_abnormal = is_abnormal_category(category_name, config)
        category_prompt = load_category_skill(category_name, skills_dir)
        logger.debug("开始处理类别: %s (is_abnormal=%s)", category_name, is_abnormal)
        _, raw_content, result, trace = await client.analyze_category_sequential(
            video_url=video_url,
            category_name=category_name,
            category_prompt=category_prompt,
            previous_preds=inherited_preds,
            output_dir=str(video_output_dir),
            is_abnormal=is_abnormal,
            detailed=detailed,
            video_duration=video_duration,
        )
        raw_file = video_output_dir / f"{category_name}_raw.txt"
        raw_file.write_text(raw_content, encoding="utf-8")
        trace_file = video_output_dir / f"{category_name}_trace.json"
        trace_file.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
        if detailed:
            logger.debug("==== [%s] 详细追踪 ====", category_name)
            logger.debug(json.dumps(trace, ensure_ascii=False, indent=2))
            logger.debug("==== [%s] 最终解析 ====\n%s", category_name, json.dumps(result, ensure_ascii=False, indent=2))
        return category_name, result

    valid_categories: list[str] = []
    for category_name in category_order:
        if category_name not in categories:
            logger.warning("跳过不在 categories.json 中的类别: %s", category_name)
            continue
        valid_categories.append(category_name)

    if category_execution_mode == "serial":
        for category_name in valid_categories:
            cat, result = await _run_one_category(category_name, previous_preds)
            parsed_results[cat] = result
            if not ResponseParser.is_parse_failed(result):
                previous_preds[cat] = result.get("pred", [])
            else:
                logger.warning("类别 %s 解析失败，不将其结果传递给后续类别", cat)
            logger.debug("完成类别: %s -> pred=%s", cat, result.get("pred", []))
    else:
        semaphore = asyncio.Semaphore(max(1, category_concurrency))

        async def _run_with_limit(category_name: str):
            async with semaphore:
                return await _run_one_category(category_name, {})

        tasks = [asyncio.create_task(_run_with_limit(cat)) for cat in valid_categories]
        results = await asyncio.gather(*tasks)
        results_map = {cat: result for cat, result in results}
        for cat in valid_categories:
            result = results_map[cat]
            parsed_results[cat] = result
            logger.debug("完成类别: %s -> pred=%s", cat, result.get("pred", []))

    result_file = video_output_dir / "result.json"
    result_file.write_text(
        json.dumps(parsed_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return output_key, parsed_results


async def main() -> None:
    args = parse_args()
    configure_runtime_logging(args.test)

    # --resume: load config from the existing output directory
    if args.resume:
        resume_dir = Path(args.resume)
        if not resume_dir.exists():
            raise FileNotFoundError(f"续传目录不存在: {resume_dir}")
        resume_config_file = resume_dir / "config.yaml"
        if not resume_config_file.exists():
            raise FileNotFoundError(f"续传目录中缺少 config.yaml: {resume_config_file}")
        config = load_config(str(resume_config_file))
        logger.info("断点续传模式：从 %s 加载配置", resume_config_file)
    else:
        config = load_config(args.config)

    config = _apply_overrides(config, args.set)
    config["_git_commit"] = get_git_commit()
    logger.info("使用配置文件: %s | git commit: %s",
                args.resume or args.config, config["_git_commit"])
    logger.info(
        "API 目标: %s | model_name: %s",
        config["api"]["base_url"],
        config["api"]["model_name"],
    )
    if args.dry_run:
        logger.info("dry-run 模式：仅校验并打印配置，不执行视频处理")
        return

    skills_dir = config["paths"].get("skills", "skills")
    init_prompts(
        config["paths"]["categories"],
        skills_dir,
    )

    client = QwenClient(config)
    client.init_system_prompt(build_global_system_prompt(skills_dir))
    client.init_split_system_prompts(
        simple_prompt=build_simple_system_prompt(skills_dir),
        complex_prompt=build_complex_system_prompt(skills_dir),
    )

    # Determine output directory
    if args.resume:
        output_dir = Path(args.resume)
        logger.info("断点续传输出目录: %s", output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(config["paths"]["output_base"]) / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        save_config(config, output_dir)
        logger.info("输出目录: %s", output_dir)

    video_root = config["paths"]["video_root"]
    video_paths_file = config["paths"].get("video_paths")
    if video_paths_file and Path(video_paths_file).exists():
        annotations = load_video_paths(video_paths_file, video_root)
        logger.info("从 video_paths 文件加载 %d 条视频记录: %s", len(annotations), video_paths_file)
    else:
        annotations = load_annotations(config["paths"]["annotations"])
        logger.info("从 annotations 文件加载 %d 条视频记录", len(annotations))

    video_semaphore = asyncio.Semaphore(config["concurrency"]["max_concurrent_videos"])
    summary_lock = asyncio.Lock()  # Protect summary.json writes from concurrent conflicts
    if args.video:
        normalized = _normalize_video_path(args.video, video_root)
        if not normalized:
            raise ValueError(f"--video 路径不在 video_root 下或无效: {args.video}")
        annotations = [{"video_path": normalized}]
        logger.info("单视频模式：%s", normalized)
    elif args.test:
        annotations = annotations[:1]
        logger.info("测试模式：自动选择首个视频进行全流程验证")
    else:
        max_clips = config["test"]["max_clips"]
        if max_clips and max_clips > 0:
            annotations = annotations[:max_clips]
            logger.info("测试配置：只处理前 %d 个视频", max_clips)

    # Resume: filter out already-completed videos
    summary: dict = {}
    if args.resume:
        annotations, summary = scan_completed_videos(output_dir, annotations, video_root)
        if not annotations:
            logger.info("所有视频均已完成，无需继续处理")
            await client.close()
            return

    total = len(annotations)

    summary_file = output_dir / "summary.json"
    if not args.resume:
        async with summary_lock:
            write_json_atomic(summary_file, summary)

    pbar = tqdm(total=total, desc="处理视频", unit="个", dynamic_ncols=True, file=sys.stdout)

    use_langgraph_by_default = bool(config.get("pipeline", {}).get("default_use_langgraph", True))
    use_langgraph = bool(args.use_langgraph or use_langgraph_by_default)
    if args.legacy_pipeline:
        use_langgraph = False

    # Determine agent mode
    agent_mode = args.agent_mode or config.get("pipeline", {}).get("agent_mode", "dag")
    if agent_mode == "react":
        use_langgraph = True  # react mode uses LangGraph

    if use_langgraph and agent_mode == "react":
        logger.info("当前执行模式：ReAct 智能体流水线（新）")
    elif use_langgraph:
        logger.info("当前执行模式：LangGraph 多智能体流水线")
    else:
        logger.info("当前执行模式：原始串并行流水线（未启用 supervisor）")

    # Interactive printer for single video mode
    show_thinking = args.show_thinking or bool(config.get("react_agent", {}).get("show_thinking", False))
    interactive_printer = None
    if (args.video or args.test) and (agent_mode == "react" or show_thinking):
        interactive_printer = InteractivePrinter(show_thinking=show_thinking)

    try:
        if use_langgraph and agent_mode == "react":
            # Use ReAct agent pipeline
            logger.info("使用 ReAct 智能体流水线")

            async def process_react(annotation: dict) -> tuple[str, dict]:
                async with video_semaphore:
                    video_path = annotation["video_path"]
                    result = await run_react_video_analysis(
                        video_path=video_path,
                        video_root=video_root,
                        output_dir=str(output_dir),
                        qwen_client=client,
                        config=config,
                        skills_dir=skills_dir,
                        previous_preds=None,
                        on_step=interactive_printer,
                    )
                    output_key = build_output_key(video_root, video_path)
                    return output_key, result

            pending = {
                asyncio.ensure_future(process_react(ann)): ann
                for ann in annotations
            }

            for coro in asyncio.as_completed(list(pending.keys())):
                try:
                    output_key, results = await coro
                    _save_per_video_result(output_dir, output_key, results)
                    async with summary_lock:
                        summary[output_key] = build_summary_item(
                            results,
                            config,
                            video_duration=_get_video_duration(output_key),
                        )
                        write_json_atomic(summary_file, summary)
                except Exception as exc:
                    logger.error("视频处理异常: %s", exc)
                finally:
                    pbar.update(1)
        elif use_langgraph:
            # Use new LangGraph multi-agent pipeline
            logger.info("使用 LangGraph 多智能体流水线")

            async def process_langgraph(annotation: dict) -> tuple[str, dict]:
                async with video_semaphore:
                    video_path = annotation["video_path"]
                    result = await run_video_analysis(
                        video_path=video_path,
                        video_root=video_root,
                        output_dir=str(output_dir),
                        qwen_client=client,
                        config=config,
                        skills_dir=skills_dir,
                        previous_preds=None,
                    )
                    output_key = build_output_key(video_root, video_path)
                    return output_key, result

            pending = {
                asyncio.ensure_future(process_langgraph(ann)): ann
                for ann in annotations
            }

            for coro in asyncio.as_completed(list(pending.keys())):
                try:
                    output_key, results = await coro
                    _save_per_video_result(output_dir, output_key, results)
                    async with summary_lock:
                        summary[output_key] = build_summary_item(
                            results,
                            config,
                            video_duration=_get_video_duration(output_key),
                        )
                        write_json_atomic(summary_file, summary)
                except Exception as exc:
                    logger.error("视频处理异常: %s", exc)
                finally:
                    pbar.update(1)
        else:
            # Use original pipeline
            async def process_with_limit(annotation: dict) -> tuple[str, dict]:
                async with video_semaphore:
                    return await process_video(
                        annotation,
                        output_dir,
                        client,
                        config,
                        detailed=args.test or bool(args.video),
                    )

            pending = {
                asyncio.ensure_future(process_with_limit(ann)): ann
                for ann in annotations
            }

            for coro in asyncio.as_completed(list(pending.keys())):
                try:
                    output_key, results = await coro
                    _save_per_video_result(output_dir, output_key, results)
                    async with summary_lock:
                        summary[output_key] = build_summary_item(
                            results,
                            config,
                            video_duration=_get_video_duration(output_key),
                        )
                        write_json_atomic(summary_file, summary)
                except Exception as exc:
                    logger.error("视频处理异常: %s", exc)
                finally:
                    pbar.update(1)
    finally:
        pbar.close()
        async with summary_lock:
            write_json_atomic(summary_file, summary)
        await client.close()

    logger.info("全部处理完成，共 %d 个视频，结果保存在: %s", len(summary), output_dir)


if __name__ == "__main__":
    asyncio.run(main())
