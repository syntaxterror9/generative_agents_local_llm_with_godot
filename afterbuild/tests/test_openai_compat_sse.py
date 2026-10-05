import pytest

from backends.base import BackendError
from backends.openai_compat import parse_sse_lines


async def canned(*lines: bytes):
    for line in lines:
        yield line


async def collect(*lines):
    return [chunk async for chunk in parse_sse_lines(canned(*lines))]


async def test_normal_stream_until_done():
    chunks = await collect(
        b'data: {"choices":[{"delta":{"content":"Hel"}}]}\n',
        b'\n',
        b'data: {"choices":[{"delta":{"content":"lo"}}]}\n',
        b'data: [DONE]\n',
    )
    assert chunks == ["Hel", "lo"]


async def test_role_only_and_null_content_chunks_are_skipped():
    chunks = await collect(
        b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n',
        b'data: {"choices":[{"delta":{"content":null}}]}\n',
        b'data: {"choices":[]}\n',
        b'data: {"id":"x","usage":{"total_tokens":3}}\n',
        b'data: {"choices":[{"delta":{"content":"hi"}}]}\n',
        b'data: [DONE]\n',
    )
    assert chunks == ["hi"]


async def test_comments_and_blank_lines_ignored():
    chunks = await collect(
        b': keep-alive\n',
        b'\n',
        b'data: {"choices":[{"delta":{"content":"x"}}]}\n',
        b'data: [DONE]\n',
    )
    assert chunks == ["x"]


async def test_stream_without_done_ends_normally():
    chunks = await collect(b'data: {"choices":[{"delta":{"content":"end"}}]}\n')
    assert chunks == ["end"]


async def test_malformed_json_raises():
    with pytest.raises(BackendError, match="Malformed SSE chunk"):
        await collect(b"data: {not json}\n")


async def test_error_payload_chunk_raises():
    with pytest.raises(BackendError, match="Provider error"):
        await collect(b'data: {"error":{"message":"rate limited"}}\n')


async def test_non_object_json_chunks_raise_backend_error():
    for payload in (b"data: null\n", b"data: 42\n", b'data: ["x"]\n'):
        with pytest.raises(BackendError, match="Malformed SSE chunk"):
            await collect(payload)


async def test_non_object_choice_is_skipped():
    chunks = await collect(
        b'data: {"choices":[null]}\n',
        b'data: {"choices":[{"delta":{"content":"ok"}}]}\n',
        b'data: [DONE]\n',
    )
    assert chunks == ["ok"]


async def test_non_string_content_is_skipped():
    chunks = await collect(
        b'data: {"choices":[{"delta":{"content":42}}]}\n',
        b'data: {"choices":[{"delta":null}]}\n',
        b'data: {"choices":[{"delta":{"content":"ok"}}]}\n',
        b'data: [DONE]\n',
    )
    assert chunks == ["ok"]
