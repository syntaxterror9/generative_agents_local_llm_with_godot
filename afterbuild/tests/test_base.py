import asyncio

import pytest

from backends.base import Backend, BackendError, ChatMessage, GenerationParams


class FakeBackend(Backend):
    name = "fake"
    model = "fake-1"

    async def stream_chat(self, messages, params):
        for part in ["a", "b", "c"]:
            yield part


def test_chat_accumulates_stream():
    backend = FakeBackend()
    messages = [ChatMessage(role="user", content="hi")]
    params = GenerationParams(max_tokens=10, temperature=0.7, top_p=0.9)

    assert asyncio.run(backend.chat(messages, params)) == "abc"


def test_generation_params_defaults_are_llamacpp_optional():
    params = GenerationParams(max_tokens=10, temperature=0.7, top_p=0.9)
    assert params.top_k is None
    assert params.repeat_penalty is None
    assert params.repeat_last_n is None


def test_chat_message_is_frozen():
    message = ChatMessage(role="user", content="hi")
    with pytest.raises(Exception):
        message.content = "changed"


def test_backend_error_is_exception():
    assert issubclass(BackendError, Exception)
