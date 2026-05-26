import os
import re
import json
from pathlib import Path
from typing import Optional


_CATEGORIES_CACHE: Optional[dict] = None


def _load_categories_cached() -> dict:
    global _CATEGORIES_CACHE
    if _CATEGORIES_CACHE is None:
        categories_path = Path(__file__).parent / "categories.json"
        with open(categories_path, "r", encoding="utf-8") as f:
            _CATEGORIES_CACHE = json.load(f)
    return _CATEGORIES_CACHE


class ResponseParser:

    # ------------------------------------------------------------------ #
    #  CJK quote normalization                                            #
    # ------------------------------------------------------------------ #
    _CJK_REPLACEMENTS = str.maketrans(
        {
            "\u300c": '"',  # 「 LEFT CORNER BRACKET
            "\u300d": '"',  # 」 RIGHT CORNER BRACKET
            "\u300e": '"',  # 『 LEFT WHITE CORNER BRACKET
            "\u300f": '"',  # 』 RIGHT WHITE CORNER BRACKET
            "\u201c": '"',  # " LEFT DOUBLE QUOTATION
            "\u201d": '"',  # " RIGHT DOUBLE QUOTATION
            "\u2018": "'",  # ' LEFT SINGLE QUOTATION
            "\u2019": "'",  # ' RIGHT SINGLE QUOTATION
            "\uff02": '"',  # ＂ FULLWIDTH QUOTATION
        }
    )

    @staticmethod
    def _normalize_cjk_quotes(text: str) -> str:
        """Replace CJK quotation characters with ASCII equivalents."""
        return text.translate(ResponseParser._CJK_REPLACEMENTS)

    @staticmethod
    def _strip_md_code_blocks(text: str) -> str:
        """Remove markdown code fences (```json ... ```)."""
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = re.sub(r"```\s*", "", text)
        return text

    @staticmethod
    def _strip_answer_tags(text: str) -> str:
        """Remove <answer>...</answer> wrapper tags."""
        return re.sub(r"</?answer>", "", text, flags=re.IGNORECASE).strip()

    @staticmethod
    def strip_think_tags(content: str) -> str:
        """Remove <think>...</think> and markdown code-block wrappers."""
        # 1. Remove <think>...</think> blocks
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        # Remove any orphan </think> tags left over
        content = re.sub(r"</think>", "", content)
        # 2. Remove leading markdown code fence line (```json,)
        content = re.sub(r"^\s*```(?:json)?\s*,?\s*\n?", "", content, flags=re.MULTILINE)
        # 3. Remove trailing markdown code fence
        content = re.sub(r"\n?\s*```\s*$", "", content, flags=re.MULTILINE)
        return content.strip()

    # ------------------------------------------------------------------ #
    #  JSON extraction — multi-strategy                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def extract_json(content: str) -> Optional[dict]:
        """Try multiple strategies to extract a valid JSON dict from LLM output."""
        # --- Strategy 1: direct parse (fast path) ---
        try:
            return json.loads(content)
        except (json.JSONDecodeError, ValueError):
            pass

        # --- Strategy 2: extract balanced {…} objects ---
        for candidate in ResponseParser._extract_balanced_json_objects(content):
            try:
                return json.loads(candidate)
            except (json.JSONDecodeError, ValueError):
                continue

        # --- Strategy 3: extract from <answer> tags ---
        for candidate in ResponseParser._extract_answer_json_candidates(content):
            try:
                return json.loads(candidate)
            except (json.JSONDecodeError, ValueError):
                continue

        # --- Strategy 4: CJK-quote normalization + retry ---
        normalized = ResponseParser._normalize_cjk_quotes(content)
        if normalized != content:
            try:
                return json.loads(normalized)
            except (json.JSONDecodeError, ValueError):
                pass
            for candidate in ResponseParser._extract_balanced_json_objects(normalized):
                try:
                    return json.loads(candidate)
                except (json.JSONDecodeError, ValueError):
                    continue

        return None

    @staticmethod
    def extract_json_candidates(content: str) -> list[dict]:
        """Return all parseable JSON dicts found in *content*."""
        candidates: list[dict] = []

        def _try_parse(text: str) -> None:
            try:
                obj = json.loads(text)
                if isinstance(obj, dict) and obj not in candidates:
                    candidates.append(obj)
            except (json.JSONDecodeError, ValueError):
                pass

        # Raw + normalized versions
        for text in (content, ResponseParser._normalize_cjk_quotes(content)):
            _try_parse(text)
            for candidate in ResponseParser._extract_balanced_json_objects(text):
                _try_parse(candidate)
            for candidate in ResponseParser._extract_answer_json_candidates(text):
                _try_parse(candidate)

        return candidates

    # ------------------------------------------------------------------ #
    #  Schema-aware structural extraction (last-resort fallback)          #
    # ------------------------------------------------------------------ #

    _SIMPLE_STEP_KEYS = [
        "evidence",
        "align",
        "decision",
    ]
    _COMPLEX_STEP_KEYS = [
        "step0_environment_context",
        "step1_object_detection",
        "step2_motion_analysis",
        "step3_conflict_check",
    ]

    @staticmethod
    def _structural_extract(text: str, is_abnormal: bool) -> Optional[dict]:
        """Extract known fields using the JSON schema when JSON parsing fails.

        Handles CJK quotes, missing delimiters, and other common LLM output
        quirks by leveraging knowledge of the required key names.
        """
        # Step 1: normalize text
        cleaned = ResponseParser._strip_answer_tags(text)
        cleaned = ResponseParser._strip_md_code_blocks(cleaned)
        # Remove 「」 entirely — they are emphasis marks, not JSON delimiters
        cleaned = cleaned.replace("\u300c", "").replace("\u300d", "")
        cleaned = cleaned.replace("\u300e", "").replace("\u300f", "")
        cleaned = cleaned.replace("\u201c", "").replace("\u201d", "")
        cleaned = cleaned.replace("\uff02", "")

        step_keys = (
            ResponseParser._COMPLEX_STEP_KEYS
            if is_abnormal
            else ResponseParser._SIMPLE_STEP_KEYS
        )

        result: dict = {}

        # --- extract step fields ---
        for i, key in enumerate(step_keys):
            key_pat = rf'"{re.escape(key)}"'
            key_match = re.search(key_pat, cleaned)
            if not key_match:
                continue

            colon_pos = cleaned.find(":", key_match.end())
            if colon_pos == -1 or colon_pos > key_match.end() + 10:
                continue

            value_start = colon_pos + 1

            # Find end: next known key, "pred", "events", or closing }
            end_markers = []
            for j in range(i + 1, len(step_keys)):
                end_markers.append(f'"{step_keys[j]}"')
            end_markers += ['"pred"', '"events"', "}"]

            value_end = len(cleaned)
            for marker in end_markers:
                pos = cleaned.find(marker, value_start)
                if pos != -1 and pos < value_end:
                    value_end = pos

            raw_val = cleaned[value_start:value_end]

            # Strip trailing comma, whitespace
            raw_val = raw_val.rstrip().rstrip(",").rstrip()

            # Remove surrounding quotes if both present
            if raw_val.startswith('"') and raw_val.endswith('"') and len(raw_val) > 1:
                raw_val = raw_val[1:-1]
            # Handle cases where only opening quote exists (model omitted closing)
            elif raw_val.startswith('"'):
                raw_val = raw_val[1:].rstrip().rstrip(",").rstrip('"').rstrip()

            # Unescape
            raw_val = raw_val.replace('\\"', '"').replace("\\n", "\n")
            result[key] = raw_val

        # --- extract pred ---
        pred_match = re.search(r'"pred"\s*:\s*\[([^\]]*)\]', cleaned, re.DOTALL)
        if pred_match:
            items = [
                v.strip().strip('"').strip("'")
                for v in pred_match.group(1).split(",")
                if v.strip()
            ]
            result["pred"] = items

        # --- extract events (complex categories) ---
        if is_abnormal:
            events_match = re.search(r'"events"\s*:\s*\[([\s\S]*?)\]\s*[,}\s]*$', cleaned)
            if events_match:
                events_text = "[" + events_match.group(1) + "]"
                norm_events = ResponseParser._normalize_cjk_quotes(events_text)
                try:
                    result["events"] = json.loads(norm_events)
                except (json.JSONDecodeError, ValueError):
                    result["events"] = []

        # --- validate ---
        # For structural extraction, step0_environment_context is optional
        # (backward compatibility with model outputs that don't include it)
        if is_abnormal:
            required = {"step1_object_detection", "step2_motion_analysis", "step3_conflict_check", "pred", "events"}
        else:
            required = set(ResponseParser._SIMPLE_STEP_KEYS) | {"pred"}

        if required.issubset(set(result.keys())):
            return result
        return None

    # ------------------------------------------------------------------ #
    #  Balanced / answer-tag extraction helpers                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_answer_json_candidates(content: str) -> list[str]:
        candidates: list[str] = []
        answer_blocks = re.findall(r"<answer>\s*([\s\S]*?)\s*</answer>", content, re.IGNORECASE)
        for block in answer_blocks:
            text = block.strip()
            if not text:
                continue
            candidates.append(text)
            if not text.startswith("{"):
                candidates.append("{\n" + text + "\n}")
        return candidates

    @staticmethod
    def _extract_balanced_json_objects(content: str) -> list[str]:
        candidates: list[str] = []
        stack: list[int] = []
        in_string = False
        escaped = False

        for idx, ch in enumerate(content):
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
                continue

            if ch == "{":
                stack.append(idx)
            elif ch == "}" and stack:
                start = stack.pop()
                if not stack:
                    candidates.append(content[start : idx + 1])
        return candidates

    # ------------------------------------------------------------------ #
    #  Validation                                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate_result(result: dict, is_abnormal: bool) -> bool:
        if is_abnormal:
            required_keys = {
                "step1_object_detection",
                "step2_motion_analysis",
                "step3_conflict_check",
                "pred",
                "events",
            }
        else:
            required_keys = {
                "evidence",
                "align",
                "decision",
                "pred",
            }

        if not required_keys.issubset(set(result.keys())):
            return False

        pred = result.get("pred")
        if not isinstance(pred, list):
            return False

        if is_abnormal and not isinstance(result.get("events"), list):
            return False

        return True

    @staticmethod
    def _validate_category_specific_fields(category_name: str, result: dict) -> bool:
        return True

    # ------------------------------------------------------------------ #
    #  Top-level parse                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def parse(category_name: str, raw_content: str, retry_count: int = 0, is_abnormal: bool = False) -> dict:
        content = ResponseParser.strip_think_tags(raw_content)

        # --- Phase 1: standard JSON extraction (unchanged logic) ---
        candidate_results = ResponseParser.extract_json_candidates(content)
        if not candidate_results:
            candidate_results = ResponseParser.extract_json_candidates(raw_content)

        for result in candidate_results:
            if ResponseParser._validate_result(result, is_abnormal) and ResponseParser._validate_category_specific_fields(category_name, result):
                result["category"] = category_name
                result["_retry_count"] = retry_count
                result["_parse_ok"] = True
                return result

        # --- Phase 2: schema-aware structural extraction (NEW) ---
        struct_result = ResponseParser._structural_extract(content, is_abnormal)
        if struct_result is None:
            struct_result = ResponseParser._structural_extract(raw_content, is_abnormal)
        if struct_result and ResponseParser._validate_result(struct_result, is_abnormal):
            struct_result["category"] = category_name
            struct_result["_retry_count"] = retry_count
            struct_result["_parse_ok"] = True
            struct_result["_structural_extract"] = True
            return struct_result

        # --- Fallback ---
        fallback_result = {
            "category": category_name,
            "error": "parse_failed",
            "raw": content,
            "_retry_count": retry_count,
            "_parse_ok": False,
        }
        fallback_result["pred"] = []
        if is_abnormal:
            fallback_result["events"] = []
        return fallback_result

    @staticmethod
    def is_parse_failed(result: dict) -> bool:
        if result.get("error") == "parse_failed":
            return True
        pred = result.get("pred", [])
        if isinstance(pred, list) and "其他异常情况" in pred:
            return True
        return False

    @staticmethod
    def get_default_normal_pred(category_name: str) -> list:
        categories = _load_categories_cached()
        if category_name in categories:
            options = categories[category_name]
            if options:
                return [options[0]]
        return []

    @staticmethod
    def save_repair_log(
        output_dir: str, video_url: str, category_name: str,
        attempt: int, raw_content_before_repair: str, repair_prompt: str,
    ) -> None:
        log_path = os.path.join(output_dir, "repair_log.jsonl")
        entry = {
            "attempt": attempt,
            "video_url": video_url,
            "category_name": category_name,
            "raw_content_before_repair": raw_content_before_repair,
            "repair_prompt": repair_prompt,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    @staticmethod
    def generate_retry_prompt(is_abnormal: bool = False, category_name: str = "") -> str:
        if is_abnormal:
            extra_constraint = "道路交通状况/动态交互类别：JSON 必须包含 step0_environment_context、step1_object_detection、step2_motion_analysis、step3_conflict_check、pred、events 共6个键，step0 和 step3 不能缺失。"
        else:
            extra_constraint = "普通分类：JSON 必须包含 evidence、align、decision、pred 共4个键，所有字段都不能缺失。"
        return (
            "请遵守系统提示中的输出规范重新输出。特别注意：\n"
            + extra_constraint
            + "\n---"
        )
