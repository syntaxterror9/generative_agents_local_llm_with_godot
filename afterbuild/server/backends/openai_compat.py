"""OpenAI-compatible Chat Completions client (llama-server, OpenRouter, DeepSeek)."""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, AsyncIterator, Dict, List, Tuple

import aiohttp

from .base import Backend, BackendError, ChatMessage, GenerationParams

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 60
CONNECT_TIMEOUT_S = 10


def build_request(
    profile: Dict[str, Any],
    provider_name: str,
    messages: List[ChatMessage],
    params: GenerationParams,
    stream: bool = True,
) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
    base_url = str(profile["base_url"]).rstrip("/")
    url = base_url + "/chat/completions"

    headers: Dict[str, str] = {"Accept": "text/event-stream"}
    key_env = profile.get("api_key_env")
    if key_env:
        key = os.environ.get(key_env)
        if not key:
            raise BackendError(f"Missing API key: environment variable {key_env} is not set")
        headers["Authorization"] = f"Bearer {key}"

    body: Dict[str, Any] = {
        "model": profile.get("model", provider_name),
        "messages": [{"role": m.role, "content": m.content} for m in messages],
        "stream": stream,
        "max_tokens": params.max_tokens,
        "temperature": params.temperature,
        "top_p": params.top_p,
    }
    if provider_name == "llamacpp":
        # llama-server extensions; cloud APIs reject unknown fields.
        body["cache_prompt"] = True
        if params.top_k is not None:
            body["top_k"] = params.top_k
        if params.repeat_penalty is not None:
            body["repeat_penalty"] = params.repeat_penalty
        if params.repeat_last_n is not None:
            body["repeat_last_n"] = params.repeat_last_n
    return url, headers, body


async def parse_sse_lines(lines: AsyncIterator[bytes]) -> AsyncIterator[str]:
    """Yield content deltas from an OpenAI-style SSE byte stream."""
    async for raw in lines:
        line = raw.decode("utf-8", errors="replace").strip()
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            return
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError as exc:
            raise BackendError(f"Malformed SSE chunk: {data[:120]!r}") from exc
        if not isinstance(chunk, dict):
            raise BackendError(f"Malformed SSE chunk: {data[:120]!r}")
        if "error" in chunk:
            raise BackendError(f"Provider error: {str(chunk['error'])[:200]}")
        choices = chunk.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            continue
        delta = choices[0].get("delta")
        if not isinstance(delta, dict):
            continue
        content = delta.get("content")
        if isinstance(content, str) and content:
            yield content


class OpenAICompatBackend(Backend):
    """SSE client for any OpenAI-compatible /chat/completions endpoint."""

    def __init__(self, name: str, profile: Dict[str, Any], config: Dict[str, Any]):
        self.name = name
        self.profile = profile
        self.config = config
        self.model = str(profile.get("model", name))
        self._session: aiohttp.ClientSession | None = None

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(
                total=float(self.config.get("request_timeout_s", DEFAULT_TIMEOUT_S)),
                connect=CONNECT_TIMEOUT_S,
            )
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def stream_chat(
        self, messages: List[ChatMessage], params: GenerationParams
    ) -> AsyncIterator[str]:
        url, headers, body = build_request(
            self.profile, self.name, messages, params
        )
        session = self._get_session()
        try:
            async with session.post(url, json=body, headers=headers) as response:
                if response.status != 200:
                    text = (await response.text())[:200]
                    raise BackendError(
                        f"{self.name} returned HTTP {response.status}: {text}"
                    )
                async for content in parse_sse_lines(response.content):
                    yield content
        except aiohttp.ClientError as exc:
            raise BackendError(f"{self.name} request failed: {exc}") from exc
        except asyncio.TimeoutError as exc:
            raise BackendError(f"{self.name} request timed out") from exc

    async def startup_check(self) -> None:
        if self.name != "llamacpp":
            return
        base_url = str(self.profile["base_url"]).rstrip("/")
        root = base_url[:-3] if base_url.endswith("/v1") else base_url
        health_url = root + "/health"
        try:
            session = self._get_session()
            async with session.get(
                health_url, timeout=aiohttp.ClientTimeout(total=3)
            ) as response:
                if response.status == 200:
                    logger.info("llama-server health OK at %s", health_url)
                else:
                    logger.warning(
                        "llama-server health check: HTTP %s at %s",
                        response.status, health_url,
                    )
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            logger.warning(
                "llama-server not reachable at %s (%s); server will still start",
                health_url, exc,
            )

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
