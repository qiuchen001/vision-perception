"""
Tool Registry for ReAct Agent.

Provides a unified interface for defining, registering, and dispatching tools.
Supports both OpenAI native function calling and XML-based fallback.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result from a tool execution."""
    output: str = ""
    media_blocks: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class ToolDefinition:
    """A single tool with OpenAI-compatible schema and async executor."""
    name: str
    description: str
    parameters: dict  # JSON Schema
    executor: Callable[..., Awaitable[ToolResult]]

    def to_openai_schema(self) -> dict:
        """Convert to OpenAI function calling schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }

    def to_xml_description(self) -> str:
        """Convert to XML-formatted description for prompt injection fallback."""
        props = self.parameters.get("properties", {})
        required = self.parameters.get("required", [])
        param_lines = []
        for pname, pdef in props.items():
            ptype = pdef.get("type", "any")
            desc = pdef.get("description", "")
            req = " (必需)" if pname in required else ""
            enum_vals = pdef.get("enum")
            enum_str = f", 可选值: {enum_vals}" if enum_vals else ""
            default_val = pdef.get("default")
            default_str = f", 默认: {default_val}" if default_val is not None else ""
            param_lines.append(f"  - {pname}: {ptype}{req}{enum_str}{default_str} — {desc}")
        params_text = "\n".join(param_lines) if param_lines else "  (无参数)"
        return (
            f"### {self.name}\n"
            f"{self.description}\n\n"
            f"参数:\n{params_text}\n\n"
            f"调用格式:\n<tool>\n"
            f'{{"tool":"{self.name}","arguments":{{"param":"value"}}}}\n'
            f"</tool>"
        )


class ToolRegistry:
    """Registry that collects, validates, and dispatches tool calls."""

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s", tool.name)

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def get_tool(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def get_openai_tools_param(self, tool_names: list[str] | None = None) -> list[dict]:
        """Build the `tools` parameter for OpenAI chat.completions.create()."""
        names = tool_names or list(self._tools.keys())
        return [
            self._tools[n].to_openai_schema()
            for n in names
            if n in self._tools
        ]

    def get_xml_tool_descriptions(self, tool_names: list[str] | None = None) -> str:
        """Build XML-formatted tool descriptions for prompt injection."""
        names = tool_names or list(self._tools.keys())
        parts = [
            self._tools[n].to_xml_description()
            for n in names
            if n in self._tools
        ]
        if not parts:
            return ""
        return "# 可用工具\n\n" + "\n\n".join(parts)

    async def dispatch(self, tool_name: str, arguments: dict, context: dict) -> ToolResult:
        """Execute a tool and return its result."""
        tool = self._tools.get(tool_name)
        if tool is None:
            return ToolResult(output=f"错误：未知工具 '{tool_name}'")
        try:
            result = await tool.executor(arguments, context)
            if not isinstance(result, ToolResult):
                result = ToolResult(output=str(result))
            return result
        except Exception as e:
            logger.error("Tool %s execution error: %s", tool_name, e)
            return ToolResult(output=f"工具执行错误 [{tool_name}]: {e}")

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())


def parse_native_tool_calls(message) -> list[dict]:
    """Parse tool calls from OpenAI native function calling response.

    Returns list of {"tool": name, "arguments": dict}.
    """
    tool_calls = getattr(message, "tool_calls", None)
    if not tool_calls:
        return []
    parsed = []
    for tc in tool_calls:
        func = getattr(tc, "function", None)
        if not func:
            continue
        name = getattr(func, "name", "")
        args_str = getattr(func, "arguments", "")
        if not name:
            continue
        try:
            args = json.loads(args_str) if args_str else {}
        except json.JSONDecodeError:
            args = {}
        parsed.append({"tool": name, "arguments": args})
    return parsed


def parse_xml_tool_calls(content: str) -> list[dict]:
    """Parse tool calls from content (XML <tool> tags or raw JSON in content).

    Handles multiple formats from different models:
    - <tool>{"tool": "xxx", "arguments": {...}}</tool>
    - {"tool": "xxx", "arguments": {...}} (no XML wrapper)
    - {"name": "xxx", "arguments": {...}} / {"name": "xxx", "parameters": {...}}
    - {"function": {"name": "xxx", "arguments": "..."}}

    Returns list of {"tool": name, "arguments": dict}.
    """
    import re

    KNOWN_TOOLS = {
        "clip_select", "frame_extract", "yolo_detect",
        "query_category_result", "get_video_info",
        "spawn_sub_agent", "review_result",
    }

    # Strip thinking tags first
    content_clean = re.sub(r"</?think>.*?</think>", "", content, flags=re.DOTALL).strip()

    # Strategy 1: Extract from <tool> XML tags
    tool_texts: list[str] = []
    for match in re.finditer(r"<tool>(.*?)</tool>", content_clean, re.DOTALL):
        tool_texts.append(match.group(1).strip())

    # Strategy 2: If no <tool> tags, scan the entire content for JSON tool calls
    if not tool_texts:
        tool_texts = [content_clean]

    results = []
    for tool_text in tool_texts:
        # Try full JSON parse of the whole text
        try:
            parsed = json.loads(tool_text)
            r = _normalize_tool_call_dict(parsed, KNOWN_TOOLS)
            if r:
                results.append(r)
                continue
            if isinstance(parsed, list):
                for item in parsed:
                    r2 = _normalize_tool_call_dict(item, KNOWN_TOOLS)
                    if r2:
                        results.append(r2)
                continue
        except json.JSONDecodeError:
            pass

        # Extract all balanced-brace JSON objects and try each
        for start in range(len(tool_text)):
            if tool_text[start] != "{":
                continue
            depth = 0
            for idx in range(start, len(tool_text)):
                if tool_text[idx] == "{":
                    depth += 1
                elif tool_text[idx] == "}":
                    depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(tool_text[start:idx + 1])
                        r = _normalize_tool_call_dict(obj, KNOWN_TOOLS)
                        if r:
                            results.append(r)
                    except json.JSONDecodeError:
                        pass
                    break

    return results


def _normalize_tool_call_dict(obj: dict, known_tools: set) -> dict | None:
    """Normalize various tool call dict formats to {"tool": name, "arguments": dict}.

    Handles:
    - {"tool": "xxx", "arguments": {...}}
    - {"name": "xxx", "arguments": {...}}
    - {"name": "xxx", "parameters": {...}}
    - {"function": {"name": "xxx", "arguments": "..."}}
    - {"category": "xxx", ...} where we infer spawn_sub_agent
    """
    if not isinstance(obj, dict):
        return None

    # Direct format: {"tool": "xxx", "arguments": {...}}
    tool_name = obj.get("tool") or obj.get("name")
    if tool_name and isinstance(tool_name, str):
        args = obj.get("arguments") or obj.get("parameters") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if not isinstance(args, dict):
            args = {}
        return {"tool": tool_name, "arguments": args}

    # OpenAI function calling format: {"function": {"name": "xxx", "arguments": "..."}}
    func = obj.get("function")
    if isinstance(func, dict):
        func_name = func.get("name", "")
        func_args = func.get("arguments", "{}")
        if isinstance(func_args, str):
            try:
                func_args = json.loads(func_args)
            except json.JSONDecodeError:
                func_args = {}
        if func_name:
            return {"tool": func_name, "arguments": func_args if isinstance(func_args, dict) else {}}

    # Infer spawn_sub_agent from {"category": "xxx", ...}
    category = obj.get("category")
    if category and not tool_name:
        # Remove category from args to avoid duplication
        args = {k: v for k, v in obj.items() if k not in ("tool", "name", "category") and v}
        if category not in args:
            args["category"] = category
        return {"tool": "spawn_sub_agent", "arguments": args}

    return None
