"""No-model backend for protocol tests."""
from __future__ import annotations

from typing import AsyncIterator, List

from .base import Backend, ChatMessage, GenerationParams


class EchoBackend(Backend):
    name = "echo"
    model = "echo"

    async def stream_chat(
        self, messages: List[ChatMessage], params: GenerationParams
    ) -> AsyncIterator[str]:
        last_user = ""
        for message in reversed(messages):
            if message.role == "user":
                last_user = message.content
                break
        text = "Echo: " + last_user
        for index, word in enumerate(text.split(" ")):
            yield word if index == 0 else " " + word
