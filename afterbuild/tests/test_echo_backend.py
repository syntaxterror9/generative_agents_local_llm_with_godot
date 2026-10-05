import pytest

from backends.base import ChatMessage, GenerationParams
from backends.echo_backend import EchoBackend

PARAMS = GenerationParams(max_tokens=50, temperature=0.0, top_p=1.0)


async def test_streams_last_user_message_as_words():
    backend = EchoBackend()
    messages = [
        ChatMessage(role="system", content="sys"),
        ChatMessage(role="user", content="first"),
        ChatMessage(role="assistant", content="reply"),
        ChatMessage(role="user", content="hello world"),
    ]
    deltas = [d async for d in backend.stream_chat(messages, PARAMS)]
    assert "".join(deltas) == "Echo: hello world"
    assert len(deltas) > 1  # actually streamed, not one blob


async def test_chat_accumulates():
    backend = EchoBackend()
    result = await backend.chat([ChatMessage(role="user", content="hi")], PARAMS)
    assert result == "Echo: hi"


async def test_no_user_message_yields_empty_echo():
    backend = EchoBackend()
    result = await backend.chat([ChatMessage(role="system", content="sys")], PARAMS)
    assert result == "Echo: "
