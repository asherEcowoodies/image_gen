from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_model: str
    openai_timeout_seconds: float
    openai_max_retries: int
    kie_api_key: str
    kie_base_url: str
    kie_model: str
    kie_aspect_ratio: str
    kie_resolution: str
    kie_output_format: str
    kie_poll_interval_seconds: float
    kie_timeout_seconds: float
    kie_max_retries: int


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise RuntimeError(f"Invalid float for {name}: {raw}") from exc


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"Invalid int for {name}: {raw}") from exc


@lru_cache
def get_settings() -> Settings:
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    kie_api_key = os.getenv("KIE_API_KEY", "").strip()

    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required")
    if not kie_api_key:
        raise RuntimeError("KIE_API_KEY is required")

    return Settings(
        openai_api_key=openai_api_key,
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        openai_timeout_seconds=_get_float("OPENAI_TIMEOUT_SECONDS", 45.0),
        openai_max_retries=max(1, _get_int("OPENAI_MAX_RETRIES", 3)),
        kie_api_key=kie_api_key,
        kie_base_url=os.getenv("KIE_BASE_URL", "https://api.kie.ai/api/v1").rstrip("/"),
        kie_model=os.getenv("KIE_MODEL", "nano-banana-2"),
        kie_aspect_ratio=os.getenv("KIE_ASPECT_RATIO", "1:1"),
        kie_resolution=os.getenv("KIE_RESOLUTION", "1K"),
        kie_output_format=os.getenv("KIE_OUTPUT_FORMAT", "jpg"),
        kie_poll_interval_seconds=max(1.0, _get_float("KIE_POLL_INTERVAL_SECONDS", 3.0)),
        kie_timeout_seconds=max(15.0, _get_float("KIE_TIMEOUT_SECONDS", 180.0)),
        kie_max_retries=max(1, _get_int("KIE_MAX_RETRIES", 3)),
    )
