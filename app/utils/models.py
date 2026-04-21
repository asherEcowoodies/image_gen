from __future__ import annotations

import re
from typing import List

from pydantic import BaseModel, Field, field_validator

_HASHTAG_RE = re.compile(r"[^a-zA-Z0-9_#]+")


class GenerateRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=4000)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        normalized = " ".join(value.split()).strip()
        if not normalized:
            raise ValueError("query cannot be empty")
        return normalized


class Captions(BaseModel):
    linkedin: str = Field(..., min_length=40, max_length=2500)
    instagram: str = Field(..., min_length=20, max_length=2200)
    twitter: str = Field(..., min_length=10, max_length=280)
    facebook: str = Field(..., min_length=20, max_length=2500)
    pinterest: str = Field(..., min_length=20, max_length=1000)

    @field_validator("*")
    @classmethod
    def clean_caption(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("caption cannot be empty")
        return normalized


class AgentResult(BaseModel):
    image_prompt: str = Field(..., min_length=80, max_length=12000)
    captions: Captions
    hashtags: List[str] = Field(..., min_length=5, max_length=30)

    @field_validator("image_prompt")
    @classmethod
    def clean_image_prompt(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("image_prompt cannot be empty")
        return normalized

    @field_validator("hashtags")
    @classmethod
    def normalize_hashtags(cls, tags: List[str]) -> List[str]:
        normalized: list[str] = []
        seen: set[str] = set()

        for tag in tags:
            cleaned = _HASHTAG_RE.sub("", tag.strip().replace(" ", ""))
            if not cleaned:
                continue
            if not cleaned.startswith("#"):
                cleaned = f"#{cleaned}"
            dedupe_key = cleaned.lower()
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            normalized.append(cleaned)

        if len(normalized) < 5:
            raise ValueError("at least 5 unique hashtags are required")

        return normalized[:30]


class GenerateResponse(BaseModel):
    image_prompt: str
    captions: Captions
    hashtags: List[str]
    image_url: str
