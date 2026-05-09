"""
Robust JSON extraction from LLM responses.
Handles: pure JSON, markdown code fences, prose + JSON, nested fences.
"""
from __future__ import annotations

import json
import re


def extract_json(text: str) -> dict:
    text = text.strip()

    # 1. Try direct parse first (pure JSON response)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown code fences ```json ... ``` or ``` ... ```
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # 3. Extract first {...} block from anywhere in the text
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            # Try to find the largest valid JSON object
            for match in re.finditer(r"\{", text):
                start = match.start()
                depth = 0
                for i, ch in enumerate(text[start:], start):
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            try:
                                return json.loads(text[start : i + 1])
                            except json.JSONDecodeError:
                                break

    raise ValueError(f"No valid JSON found in LLM response: {text[:200]!r}")
