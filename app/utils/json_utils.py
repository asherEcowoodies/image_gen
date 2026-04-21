from __future__ import annotations

import json
import re
from typing import Any

_URL_PATTERN = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)


def extract_json_object(raw_text: str) -> dict[str, Any]:
    raw_text = raw_text.strip()
    if not raw_text:
        raise ValueError("Empty response from model")

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No valid JSON object found")
        parsed = json.loads(raw_text[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("Model output must be a JSON object")

    return parsed


def extract_first_url(value: Any) -> str | None:
    if value is None:
        return None

    if isinstance(value, str):
        match = _URL_PATTERN.search(value)
        return match.group(0) if match else None

    if isinstance(value, list):
        for item in value:
            found = extract_first_url(item)
            if found:
                return found
        return None

    if isinstance(value, dict):
        for item in value.values():
            found = extract_first_url(item)
            if found:
                return found
        return None

    return None
