from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any

import httpx

from app.services.exceptions import KieServiceError, KieTaskFailedError, KieTimeoutError
from app.utils.config import Settings
from app.utils.json_utils import extract_first_url

_TRANSIENT_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
_FAILURE_STATES = {"failed", "error", "cancelled", "canceled", "timeout"}
_SUCCESS_STATES = {"success", "succeeded", "completed", "done"}
_NON_ERROR_CODES = {"0", "200", "ok", "success"}
_BEARER_PREFIX_RE = re.compile(r"^\s*bearer\s+", re.IGNORECASE)


class _TransientHTTPError(Exception):
    pass


def _normalize_api_key(raw_key: str) -> str:
    # Accept keys pasted as either plain token or "Bearer <token>".
    return _BEARER_PREFIX_RE.sub("", raw_key).strip()


def _get_task_id(payload: dict[str, Any]) -> str | None:
    data = payload.get("data")
    data_dict = data if isinstance(data, dict) else {}

    candidates = (
        data_dict.get("taskId"),
        data_dict.get("task_id"),
        data_dict.get("id"),
        payload.get("taskId"),
        payload.get("task_id"),
        payload.get("id"),
    )
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _compact_payload(payload: Any, limit: int = 400) -> str:
    try:
        text = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    except Exception:
        text = str(payload)
    return text[:limit]


class KieService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=15.0,
                read=settings.kie_timeout_seconds,
                write=15.0,
                pool=15.0,
            )
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def generate_image(self, image_prompt: str) -> str:
        task_id = await self._create_task(image_prompt=image_prompt)
        return await self._poll_task(task_id=task_id)

    async def _create_task(self, image_prompt: str) -> str:
        payload = {
            "model": self._settings.kie_model,
            "input": {
                "prompt": image_prompt,
                "aspect_ratio": self._settings.kie_aspect_ratio,
                "resolution": self._settings.kie_resolution,
                "output_format": self._settings.kie_output_format,
            },
        }

        response = await self._request_json(
            method="POST",
            url=f"{self._settings.kie_base_url}/jobs/createTask",
            json=payload,
        )

        task_id = _get_task_id(response)

        if not task_id:
            raise KieServiceError(
                "Kie createTask did not return a valid task id. "
                f"Response: {_compact_payload(response)}"
            )

        return task_id

    async def _poll_task(self, task_id: str) -> str:
        deadline = time.monotonic() + self._settings.kie_timeout_seconds

        while time.monotonic() < deadline:
            response = await self._request_json(
                method="GET",
                url=f"{self._settings.kie_base_url}/jobs/recordInfo",
                params={"taskId": task_id},
            )

            if not isinstance(response, dict):
                raise KieServiceError("Kie recordInfo returned an invalid response body")

            data = response.get("data")
            if not isinstance(data, dict):
                data = response

            state = str(
                data.get("state") or data.get("status") or data.get("taskStatus") or ""
            ).strip().lower()
            if state in _SUCCESS_STATES:
                image_url = (
                    extract_first_url(data.get("resultJson"))
                    or extract_first_url(data.get("result"))
                    or extract_first_url(data)
                )
                if not image_url:
                    raise KieServiceError("Kie task succeeded but image URL was not found")
                return image_url

            if state in _FAILURE_STATES:
                raise KieTaskFailedError(f"Kie task failed with state: {state}")

            await asyncio.sleep(self._settings.kie_poll_interval_seconds)

        raise KieTimeoutError(
            f"Timed out waiting for Kie task {task_id} after {self._settings.kie_timeout_seconds}s"
        )

    async def _request_json(self, method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        headers = kwargs.pop("headers", {})
        headers["Authorization"] = f"Bearer {_normalize_api_key(self._settings.kie_api_key)}"
        headers.setdefault("Content-Type", "application/json")

        for attempt in range(1, self._settings.kie_max_retries + 1):
            try:
                response = await self._client.request(method, url, headers=headers, **kwargs)
                if response.status_code in _TRANSIENT_STATUS_CODES:
                    raise _TransientHTTPError(f"Transient HTTP {response.status_code}")
                if response.status_code >= 400:
                    raise KieServiceError(
                        f"Kie request failed ({response.status_code}): {response.text[:400]}"
                    )

                try:
                    payload = response.json()
                except ValueError as exc:
                    raise KieServiceError("Kie response was not valid JSON") from exc

                if not isinstance(payload, dict):
                    raise KieServiceError("Kie response JSON root must be an object")

                if "code" in payload:
                    code = str(payload.get("code", "")).strip().lower()
                    if code and code not in _NON_ERROR_CODES:
                        msg = payload.get("msg") or payload.get("message") or payload.get("error")
                        raise KieServiceError(
                            f"Kie API returned code={payload.get('code')}: {msg}"
                        )

                return payload
            except (httpx.RequestError, _TransientHTTPError) as exc:
                if attempt >= self._settings.kie_max_retries:
                    raise KieServiceError(f"Kie request retry exhausted: {exc}") from exc
                await asyncio.sleep(min(2 ** (attempt - 1), 8))

        raise KieServiceError("Kie request failed after retries")
