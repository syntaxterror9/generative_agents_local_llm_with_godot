"""Provider-agnostic chat backend interface."""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import AsyncIterator, List


class BackendError(Exception):
    """Any provider failure: transport, HTTP status, malformed stream, missing key."""


@dataclass(frozen=True)
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass(frozen=True)
class GenerationParams:
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int | None = None             # llama.cpp only; ignored by cloud providers
    repeat_penalty: float | None = None  # llama.cpp only
    repeat_last_n: int | None = None     # llama.cpp only


class Backend(abc.ABC):
    """One primitive: stream content deltas for a full message list."""

    name: str = "backend"
    model: str = ""

    @abc.abstractmethod
    def stream_chat(
        self, messages: List[ChatMessage], params: GenerationParams
    ) -> AsyncIterator[str]:
        """Implement as ``async def`` with ``yield``. Raise BackendError on failure."""

    async def chat(self, messages: List[ChatMessage], params: GenerationParams) -> str:
        parts: List[str] = []
        async for delta in self.stream_chat(messages, params):
            parts.append(delta)
        return "".join(parts)

    async def startup_check(self) -> None:
        """Optional connectivity probe. Logs; never raises."""

    async def close(self) -> None:
        """Release resources."""
