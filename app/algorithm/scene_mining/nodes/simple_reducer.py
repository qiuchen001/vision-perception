"""
Simple Reducer Node for LangGraph pipeline.

Merges list of simple_preds from parallel workers into a single dict.
This is a deterministic Python reducer - no LLM involved.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def simple_reducer_node(state: dict) -> dict:
    """
    Merge simple_preds list into merged_simple dict.

    Input state keys used:
    - simple_preds OR simple_worker_batch: list[dict], each dict contains {"category": str, "pred": list}
      (Send API accumulates results to simple_worker_batch key)

    Output state keys set:
    - merged_simple: dict mapping category -> result dict
    """
    # Send API with Annotated[list, operator.add] accumulates results to simple_preds key
    simple_preds = state.get("simple_preds", [])

    merged_simple: dict[str, Any] = {}

    for pred_item in simple_preds:
        if not isinstance(pred_item, dict):
            continue
        category = pred_item.get("category")
        if not category:
            continue

        # Extract only the pred field (not full reasoning/thinking)
        merged_simple[category] = {
            "pred": pred_item.get("pred", []),
        }
        if isinstance(pred_item.get("events"), list):
            merged_simple[category]["events"] = pred_item.get("events", [])

    logger.info("Simple reducer: merged %d predictions into merged_simple", len(merged_simple))
    return {"merged_simple": merged_simple}
