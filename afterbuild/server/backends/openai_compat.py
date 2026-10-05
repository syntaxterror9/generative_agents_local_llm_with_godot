"""OpenAI-compatible Chat Completions client (llama-server, OpenRouter, DeepSeek)."""
from __future__ import annotations

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
        if "error" in chunk:
            raise BackendError(f"Provider error: {str(chunk['error'])[:200]}")
        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        content = delta.get("content")
        if content:
            yield content
