from __future__ import annotations

import asyncio
import inspect
from typing import Optional

from openai import APIError, APITimeoutError, AsyncOpenAI, RateLimitError
from pydantic import ValidationError

from app.services.exceptions import AgentOutputError
from app.utils.config import Settings
from app.utils.json_utils import extract_json_object
from app.utils.models import AgentResult

_SYSTEM_PROMPT = (
    "You are an expert content strategist, visual prompt engineer, and social media copywriter. "
    "Always write in natural Hinglish with a business coach tone. "
    "You must return only strict JSON and no extra text."
)


class ContentGenerationAgent:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.openai_timeout_seconds,
        )

    async def close(self) -> None:
        close_fn = getattr(self._client, "close", None)
        if callable(close_fn):
            maybe_result = close_fn()
            if inspect.isawaitable(maybe_result):
                await maybe_result

    async def generate(self, query: str) -> AgentResult:
        repair_feedback: Optional[str] = None
        last_error = "unknown validation error"

        for attempt in range(1, self._settings.openai_max_retries + 1):
            try:
                raw_output = await self._call_model(query=query, repair_feedback=repair_feedback)
                parsed = extract_json_object(raw_output)
                return AgentResult.model_validate(parsed)
            except (ValidationError, ValueError) as exc:
                last_error = str(exc)
                repair_feedback = (
                    "Your previous output failed validation. "
                    f"Validation error: {last_error}. "
                    "Return corrected strict JSON only."
                )
            except (RateLimitError, APITimeoutError, APIError) as exc:
                last_error = str(exc)

            if attempt < self._settings.openai_max_retries:
                await asyncio.sleep(min(2 ** (attempt - 1), 8))

        raise AgentOutputError(f"Agent failed after retries: {last_error}")

    async def _call_model(self, query: str, repair_feedback: str | None) -> str:
        user_prompt = self._build_user_prompt(query=query, repair_feedback=repair_feedback)

        response = await self._client.chat.completions.create(
            model=self._settings.openai_model,
            temperature=0,
            top_p=1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )

        content = response.choices[0].message.content if response.choices else None
        if not content:
            raise ValueError("Model returned empty content")

        return content

    @staticmethod
    def _build_user_prompt(query: str, repair_feedback: str | None) -> str:
        prompt = f"""
Input query:
{query}

Return ONLY one strict JSON object with this exact schema:
{{
  "image_prompt": "string",
  "captions": {{
    "linkedin": "string",
    "instagram": "string",
    "twitter": "string",
    "facebook": "string",
    "pinterest": "string"
  }},
  "hashtags": ["#tag1", "#tag2"]
}}

Rules:
1) Use natural Hinglish in business coach tone.
2) image_prompt must be production-ready, highly detailed, photorealistic guidance for a 1:1 social post image.
3) Captions must be platform-specific and non-repetitive.
4) twitter caption must be 280 chars or fewer.
5) hashtags must be a flat list of 15 to 25 unique relevant hashtags.
6) Do not add markdown, comments, prefixes, suffixes, or extra keys.
7) Do not include timestamps, random IDs, or unstable data.
""".strip()

        if repair_feedback:
            prompt += f"\n\nCorrection instruction:\n{repair_feedback}"

        return prompt
