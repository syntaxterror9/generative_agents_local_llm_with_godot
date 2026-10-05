import asyncio

import pytest
from aiohttp import web

from backends.base import BackendError, ChatMessage, GenerationParams
from backends.openai_compat import OpenAICompatBackend

MESSAGES = [ChatMessage(role="user", content="hello")]
PARAMS = GenerationParams(max_tokens=20, temperature=0.0, top_p=1.0)


async def health(request):
    return web.Response(text="ok")


async def ok_handler(request):
    return web.Response(text="{}")


async def start_test_server(handler):
    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    app.router.add_get("/health", health)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, port


def make_backend(port):
    profile = {"base_url": f"http://127.0.0.1:{port}/v1", "model": "test-model"}
    return OpenAICompatBackend(name="llamacpp", profile=profile, config={"request_timeout_s": 10})


async def test_streams_tokens_and_accumulates():
    async def handler(request):
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for piece in ["Hel", "lo ", "world"]:
            await response.write(f'data: {{"choices":[{{"delta":{{"content":"{piece}"}}}}]}}\n\n'.encode())
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    runner, port = await start_test_server(handler)
    backend = make_backend(port)
    try:
        deltas = [d async for d in backend.stream_chat(MESSAGES, PARAMS)]
        assert deltas == ["Hel", "lo ", "world"]
        assert await backend.chat(MESSAGES, PARAMS) == "Hello world"
    finally:
        await backend.close()
        await runner.cleanup()


async def test_non_200_raises_with_status_and_body():
    async def handler(request):
        return web.Response(status=500, text="llama-server exploded")

    runner, port = await start_test_server(handler)
    backend = make_backend(port)
    try:
        with pytest.raises(BackendError, match="500"):
            await backend.chat(MESSAGES, PARAMS)
    finally:
        await backend.close()
        await runner.cleanup()


async def test_connection_refused_raises_backend_error():
    # Bind and release a port so we know nothing is listening there.
    import socket

    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    unused_port = probe.getsockname()[1]
    probe.close()

    backend = make_backend(unused_port)
    try:
        with pytest.raises(BackendError, match="request failed"):
            await backend.chat(MESSAGES, PARAMS)
    finally:
        await backend.close()


async def test_startup_check_tolerates_dead_server():
    backend = make_backend(1)  # nothing listens on port 1
    await backend.startup_check()  # returns quietly instead of raising
    await backend.close()


async def test_close_closes_session():
    runner, port = await start_test_server(ok_handler)
    backend = make_backend(port)
    session = backend._get_session()
    await backend.close()
    assert session.closed
    await runner.cleanup()
