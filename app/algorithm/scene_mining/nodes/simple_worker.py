"""
Simple Worker Node for LangGraph pipeline.

Processes simple categories in parallel (no tool calls).
Each worker handles one category and returns only pred result.
"""

import logging

from prompts import load_category_skill, is_abnormal_category
from response_parser import ResponseParser

logger = logging.getLogger(__name__)


async def simple_worker_node(state: dict) -> dict:
    """
    Simple worker node for processing simple categories.

    Input state keys used:
    - task: dict with {"category": str, ...}
    - video_url: str
    - video_path: str
    - output_dir: str
    - qwen_client: QwenClient
    - config: dict
    - skills_dir: str

    Output state keys set:
    - simple_pred: dict with {"category": str, "pred": list}

    Note: This runs as a Send target, receiving only the task slice of the state.
    """
    task = state.get("task", {})
    category = task.get("category")

    if not category:
        logger.error("Simple worker: no category in task")
        return {"simple_pred": {"category": "unknown", "pred": []}}

    client = state["qwen_client"]
    video_url = state["video_url"]
    output_dir = state["output_dir"]
    config = state["config"]
    skills_dir = state["skills_dir"]
    on_step = state.get("on_step")
    video_duration = float(state.get("video_info", {}).get("duration", 0.0))
    if on_step:
        on_step("vlm", f"VLM分析中... {category}", {"category": category})

    # Load category prompt
    category_prompt = load_category_skill(category, skills_dir)
    is_abnormal = is_abnormal_category(category, config)

    try:
        _, raw_content, result, trace = await client.analyze_category_sequential(
            video_url=video_url,
            category_name=category,
            category_prompt=category_prompt,
            previous_preds={},  # Simple categories don't use prior context
            output_dir=output_dir,
            is_abnormal=is_abnormal,
            detailed=False,
            video_duration=video_duration,
        )

        # Extract only pred/events summary
        pred = result.get("pred", []) if isinstance(result, dict) else []
        events = result.get("events", []) if isinstance(result, dict) and is_abnormal else []

        logger.info("Simple worker completed: %s -> pred=%s", category, pred[:2] if pred else [])

        return {
            "simple_preds": [{"category": category, "pred": pred, "events": events}],
            "simple_traces": [{"category": category, "raw_content": raw_content, "trace": trace}],
            "agent_outputs": [
                {
                    "agent_name": f"simple_worker_{category}",
                    "task_name": category,
                    "raw_content": raw_content,
                    "result": {"pred": pred, "events": events},
                    "trace": trace,
                }
            ],
        }

    except Exception as e:
        logger.error("Simple worker error for %s: %s", category, e)
        default_pred = ResponseParser.get_default_normal_pred(category)
        return {
            "simple_preds": [{"category": category, "pred": default_pred if isinstance(default_pred, list) else [default_pred], "events": []}],
            "simple_traces": [{"category": category, "raw_content": "", "trace": {}}],
            "agent_outputs": [
                {
                    "agent_name": f"simple_worker_{category}",
                    "task_name": category,
                    "raw_content": "",
                    "result": {"pred": default_pred if isinstance(default_pred, list) else [default_pred], "events": []},
                    "trace": {"error": str(e)},
                }
            ],
        }
