"""
Tool system for ReAct Agent.

Provides ToolRegistry, ToolDefinition, ToolResult, and factory functions
to create pre-configured registries for root and sub agents.
"""

from tools.registry import ToolDefinition, ToolResult, ToolRegistry
from tools.clip_select import CLIP_SELECT_TOOL
from tools.frame_extract import FRAME_EXTRACT_TOOL
from tools.yolo_detect import YOLO_DETECT_TOOL
from tools.query_category_result import QUERY_CATEGORY_RESULT_TOOL
from tools.get_video_info import GET_VIDEO_INFO_TOOL
from tools.spawn_sub_agent import SPAWN_SUB_AGENT_TOOL
from tools.review_result import REVIEW_RESULT_TOOL


def create_sub_agent_registry() -> ToolRegistry:
    """Create a ToolRegistry with analysis tools for sub-agents."""
    registry = ToolRegistry()
    registry.register(CLIP_SELECT_TOOL)
    registry.register(FRAME_EXTRACT_TOOL)
    registry.register(YOLO_DETECT_TOOL)
    registry.register(QUERY_CATEGORY_RESULT_TOOL)
    registry.register(GET_VIDEO_INFO_TOOL)
    return registry


def create_root_agent_registry() -> ToolRegistry:
    """Create a ToolRegistry with management tools for root agent."""
    registry = ToolRegistry()
    registry.register(SPAWN_SUB_AGENT_TOOL)
    registry.register(QUERY_CATEGORY_RESULT_TOOL)
    registry.register(GET_VIDEO_INFO_TOOL)
    registry.register(REVIEW_RESULT_TOOL)
    return registry
