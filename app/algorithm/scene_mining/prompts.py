import json
from pathlib import Path


def load_categories(categories_path: str = "categories.json") -> dict:
    with open(categories_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_skill(skill_name: str, skills_dir: str = "skills") -> str:
    path = Path(skills_dir) / f"{skill_name}.md"
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _load_include(name: str, skills_dir: str) -> str:
    """Load a shared include file from skills/categories/."""
    path = Path(skills_dir) / "categories" / f"{name}.md"
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""


import re as _re


def load_category_skill(category_name: str, skills_dir: str = "skills") -> str:
    path = Path(skills_dir) / "categories" / f"{category_name}.md"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    for placeholder, include_name in [
        ("{_complex_common_include}", "_complex_common"),
        ("{_simple_common_include}", "_simple_common"),
    ]:
        if placeholder in content:
            content = content.replace(placeholder, _load_include(include_name, skills_dir))
    # Strip supervisor-only sections — sub-agents should not see planning guidance
    content = _re.sub(r'\n## Supervisor规划要点\n.*?(?=\n## |\Z)', '', content, flags=_re.DOTALL).strip()
    return content


def build_global_system_prompt(skills_dir: str = "skills") -> str:
    """基础 system prompt，仅包含全局规则，供所有 agent 使用。"""
    return load_skill("global_system", skills_dir)


def build_simple_system_prompt(skills_dir: str = "skills") -> str:
    """简单类别专用 system prompt。"""
    return load_skill("simple_system", skills_dir)


def build_complex_system_prompt(skills_dir: str = "skills") -> str:
    """复杂类别专用 system prompt。"""
    return load_skill("complex_system", skills_dir)




def build_complex_tool_prompt(skills_dir: str = "skills") -> str:
    """clip_select 工具说明 + 采样策略，仅注入 complex worker。"""
    return load_skill("clip_tool", skills_dir)


def build_react_system_prompt(skills_dir: str = "skills") -> str:
    """Build system prompt for the ReAct root agent.

    Supports ``{_global_system_include}`` placeholder so react_system.md
    can inherit global_system.md without duplicating content.
    """
    try:
        content = load_skill("react_system", skills_dir)
        if "{_global_system_include}" in content:
            global_system = build_global_system_prompt(skills_dir)
            content = content.replace("{_global_system_include}", global_system)
        return content
    except FileNotFoundError:
        return build_global_system_prompt(skills_dir)


def is_abnormal_category(category_name: str, config: dict) -> bool:
    return category_name in set(config.get("pipeline", {}).get("abnormal_categories", []))


def get_sorted_categories(config: dict) -> list[str]:
    return list(config.get("pipeline", {}).get("category_order", []))


_CATEGORIES: dict = {}
_SKILLS_DIR: str = "skills"


def init_prompts(categories_path: str = "categories.json", skills_dir: str = "skills") -> None:
    global _CATEGORIES, _SKILLS_DIR
    _CATEGORIES = load_categories(categories_path)
    _SKILLS_DIR = skills_dir


def get_categories() -> dict:
    return _CATEGORIES


def get_skills_dir() -> str:
    return _SKILLS_DIR
