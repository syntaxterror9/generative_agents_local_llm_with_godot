# Provider Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace GPT4All in `afterbuild/server/dialogue_server.py` with a provider layer that streams from local `llama-server` (llama.cpp), OpenRouter, or DeepSeek — all via one OpenAI-compatible HTTP client.

**Architecture:** A `backends/` package under `afterbuild/server/` defines `ChatMessage` / `GenerationParams` / a `Backend` ABC with one streaming primitive (`stream_chat`) and a shared accumulation helper (`chat`). `OpenAICompatBackend` implements it over aiohttp SSE; `EchoBackend` serves protocol tests. `dialogue_server.py` loses its in-process model, sessions, and lock: it builds a stateless `messages` list per request (system prompt + memory-derived history + new message) and streams through the backend.

**Tech Stack:** Python 3.10–3.11, aiohttp (SSE), websockets, pytest + pytest-asyncio, Godot 4.4 client.

**Spec:** `docs/superpowers/specs/2026-10-06-provider-layer-design.md`

## Global Constraints

- Working build is `afterbuild/` only; never touch `finalbuild/` (superseded, its GPT4All files were deleted 2026-10-06).
- No `gpt4all` and no `llama_cpp_python` — llama.cpp is the external `llama-server` binary only.
- One provider serves dialogue and decisions; no per-role override; no runtime switching.
- WebSocket protocol is frozen: frames `token` / `complete` / `error` / `decision_result`, always end a request with `complete` or `decision_result`, 20 ms sleep between token frames.
- New config keys must work when absent (old config files keep loading): `provider` → `"llamacpp"`, `providers` → built-in defaults, `active_context_size` → `7`, `request_timeout_s` → `60`.
- Do not change the memory JSON schema or `metadata` keys; `metadata.model` becomes `"<provider>:<model>"`.
- API keys come only from env vars (`OPENROUTER_API_KEY`, `DEEPSEEK_API_KEY`); keys must never be logged or committed.
- Python: type hints, `pathlib`, `logging` (not print) in server code.
- Commands below are run from the repo root with Windows paths; the venv python is `.venv\Scripts\python.exe`.
- Commit only the files named in each task; `CLAUDE.md` is gitignored (never `git add -f` it).

## Review Focus

Input classes / failure modes the spec implies that a reasonable user would expect to work. Each has a test added in the owning task:

1. **`delta.content` is `null`/absent** (common in role-only first chunks and tool-call chunks) → must be skipped, not crash — test in Task 4.
2. **Stream ends without `[DONE]`** (connection closed cleanly after last token) → treated as normal completion, not an error — test in Task 4.
3. **Empty or absent memory files** → message building still produces a valid `[system, user]` pair — test in Task 8.
4. **Client disconnects mid-stream** → server logs, does not die, and still serves the next connection — check in Task 8's smoke test.
5. **Error payload chunk mid-stream** (`{"error": ...}`, no `choices`) → raises `BackendError` instead of being skipped as a no-choices chunk — test in Task 4.

---

### Task 1: Backends package — base types and interface

**Files:**
- Create: `afterbuild/server/backends/__init__.py` (placeholder for now, filled in Task 6)
- Create: `afterbuild/server/backends/base.py`
- Create: `afterbuild/conftest.py`
- Create: `afterbuild/tests/test_base.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `ChatMessage(role: str, content: str)`, `GenerationParams(max_tokens, temperature, top_p, top_k=None, repeat_penalty=None, repeat_last_n=None)` (frozen dataclasses), `BackendError(Exception)`, `Backend` ABC with `name: str`, `model: str`, abstract async-generator `stream_chat(messages, params)`, and concrete `async chat(messages, params) -> str`, `async startup_check() -> None`, `async close() -> None`.

- [ ] **Step 1: Write `afterbuild/conftest.py`** so `pytest` can import from `server/`:

```python
import sys
from pathlib import Path

SERVER_DIR = Path(__file__).parent / "server"
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))
```

- [ ] **Step 2: Write the failing test** `afterbuild/tests/test_base.py`:

```python
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
    import asyncio

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
```

- [ ] **Step 3: Run it, expect failure**

Run: `.venv\Scripts\python.exe -m pytest afterbuild/tests/test_base.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backends'`

- [ ] **Step 4: Implement** `afterbuild/server/backends/base.py`:

```python
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
```

- [ ] **Step 5: Create `afterbuild/server/backends/__init__.py`** with a temporary docstring only (factory comes in Task 6):

```python
"""Inference backends for the NPC server."""
```

- [ ] **Step 6: Run the test, expect pass**

Run: `.venv\Scripts\python.exe -m pytest afterbuild/tests/test_base.py -v`
Expected: PASS (4 tests)

- [ ] **Step 7: Commit**

```bash
git add afterbuild/conftest.py afterbuild/server/backends/__init__.py afterbuild/server/backends/base.py afterbuild/tests/test_base.py
git commit -m "feat(afterbuild): add backend interface for the provider layer"
```

---

### Task 2: Echo backend

**Files:**
- Create: `afterbuild/server/backends/echo_backend.py`
- Create: `afterbuild/tests/test_echo_backend.py`

**Interfaces:**
- Consumes: `Backend`, `ChatMessage`, `GenerationParams` from Task 1.
- Produces: `EchoBackend()` with `name == "echo"`; streams words of `"Echo: <last user message>"`.

- [ ] **Step 1: Write the failing test** `afterbuild/tests/test_echo_backend.py`:

```python
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
```

- [ ] **Step 2: Run it, expect failure**

Run: `.venv\Scripts\python.exe -m pytest afterbuild/tests/test_echo_backend.py -v`
(Requires `pytest-asyncio` with `asyncio_mode = auto` — created in Task 7. If not present yet, install now: `.venv\Scripts\python.exe -m pip install pytest-asyncio`, and re-run.)
Expected: FAIL — `ModuleNotFoundError: No module named 'backends.echo_backend'`

- [ ] **Step 3: Implement** `afterbuild/server/backends/echo_backend.py`:

```python
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
```

- [ ] **Step 4: Run the test, expect pass**

Run: `.venv\Scripts\python.exe -m pytest afterbuild/tests/test_echo_backend.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add afterbuild/server/backends/echo_backend.py afterbuild/tests/test_echo_backend.py
git commit -m "feat(afterbuild): add echo backend for protocol tests"
```

---

### Task 3: Request construction for the OpenAI-compatible client

**Files:**
- Create: `afterbuild/server/backends/openai_compat.py` (first piece: `build_request`)
- Create: `afterbuild/tests/test_openai_compat_request.py`

**Interfaces:**
- Consumes: Task 1 types.
- Produces: `build_request(profile, provider_name, messages, params, stream=True) -> tuple[str, dict[str, str], dict]` returning `(url, headers, body)`. Raises `BackendError` when `profile["api_key_env"]` is set but the env var is missing. llama.cpp-only extras (`cache_prompt`, `top_k`, `repeat_penalty`, `repeat_last_n`) appear in the body **only** when `provider_name == "llamacpp"`.

- [ ] **Step 1: Write the failing test** `afterbuild/tests/test_openai_compat_request.py`:

```python
import pytest

from backends.base import BackendError, ChatMessage, GenerationParams
from backends.openai_compat import build_request

MESSAGES = [ChatMessage(role="system", content="s"), ChatMessage(role="user", content="u")]
PARAMS = GenerationParams(
    max_tokens=150, temperature=0.7, top_p=0.9,
    top_k=40, repeat_penalty=1.18, repeat_last_n=64,
)


def test_llamacpp_sends_extra_sampling_params():
    profile = {"base_url": "http://127.0.0.1:8080/v1", "model": "llama-local"}
    url, headers, body = build_request(profile, "llamacpp", MESSAGES, PARAMS)
    assert url == "http://127.0.0.1:8080/v1/chat/completions"
    assert "Authorization" not in headers
    assert body["cache_prompt"] is True
    assert body["top_k"] == 40
    assert body["repeat_penalty"] == 1.18
    assert body["repeat_last_n"] == 64
    assert body["stream"] is True
    assert body["messages"] == [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
    ]


def test_cloud_provider_gets_no_llamacpp_params(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    profile = {
        "base_url": "https://openrouter.ai/api/v1/",
        "model": "meta-llama/llama-3.2-3b-instruct",
        "api_key_env": "OPENROUTER_API_KEY",
    }
    url, headers, body = build_request(profile, "openrouter", MESSAGES, PARAMS)
    assert url == "https://openrouter.ai/api/v1/chat/completions"  # trailing slash handled
    assert headers["Authorization"] == "Bearer sk-test"
    assert "top_k" not in body and "cache_prompt" not in body
    assert body["model"] == "meta-llama/llama-3.2-3b-instruct"


def test_missing_api_key_env_raises_naming_the_variable(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    profile = {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "api_key_env": "DEEPSEEK_API_KEY",
    }
    with pytest.raises(BackendError, match="DEEPSEEK_API_KEY"):
        build_request(profile, "deepseek", MESSAGES, PARAMS)


def test_none_llamacpp_params_are_omitted():
    profile = {"base_url": "http://127.0.0.1:8080/v1", "model": "llama-local"}
    params = GenerationParams(max_tokens=10, temperature=0.1, top_p=0.5)
    _, _, body = build_request(profile, "llamacpp", MESSAGES, params)
    assert "top_k" not in body
    assert "repeat_penalty" not in body
    assert "repeat_last_n" not in body
```

- [ ] **Step 2: Run it, expect failure**

Run: `.venv\Scripts\python.exe -m pytest afterbuild/tests/test_openai_compat_request.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backends.openai_compat'`

- [ ] **Step 3: Implement** `afterbuild/server/backends/openai_compat.py` (this file grows in Tasks 4–5):

```python
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
```

- [ ] **Step 4: Run the test, expect pass**

Run: `.venv\Scripts\python.exe -m pytest afterbuild/tests/test_openai_compat_request.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add afterbuild/server/backends/openai_compat.py afterbuild/tests/test_openai_compat_request.py
git commit -m "feat(afterbuild): build OpenAI-compatible chat completion requests"
```

---

### Task 4: SSE stream parsing

**Files:**
- Modify: `afterbuild/server/backends/openai_compat.py` (append `parse_sse_lines`)
- Create: `afterbuild/tests/test_openai_compat_sse.py`

**Interfaces:**
- Consumes: `BackendError` (Task 1).
- Produces: `async parse_sse_lines(lines: AsyncIterator[bytes]) -> AsyncIterator[str]` — yields content deltas; ignores blank/comment/non-`data:` lines, chunks without `choices`, and null/empty content; stops on `data: [DONE]`; raises `BackendError` on malformed JSON or a chunk containing `"error"`.

- [ ] **Step 1: Write the failing test** `afterbuild/tests/test_openai_compat_sse.py`:

```python
import pytest

from backends.base import BackendError
from backends.openai_compat import parse_sse_lines


async def canned(*lines: bytes):
    for line in lines:
        yield line


async def collect(lines):
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
```

- [ ] **Step 2: Run it, expect failure**

Run: `.venv\Scripts\python.exe -m pytest afterbuild/tests/test_openai_compat_sse.py -v`
Expected: FAIL — `ImportError: cannot import name 'parse_sse_lines'`

- [ ] **Step 3: Append to** `afterbuild/server/backends/openai_compat.py`:

```python
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
```

- [ ] **Step 4: Run the test, expect pass**

Run: `.venv\Scripts\python.exe -m pytest afterbuild/tests/test_openai_compat_sse.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add afterbuild/server/backends/openai_compat.py afterbuild/tests/test_openai_compat_sse.py
git commit -m "feat(afterbuild): parse OpenAI-compatible SSE streams"
```

---

### Task 5: HTTP streaming backend (session, timeouts, health check)

**Files:**
- Modify: `afterbuild/server/backends/openai_compat.py` (append `OpenAICompatBackend`)
- Create: `afterbuild/tests/test_openai_compat_http.py`

**Interfaces:**
- Consumes: `build_request`, `parse_sse_lines` (Tasks 3–4); config keys `request_timeout_s` (default 60).
- Produces: `OpenAICompatBackend(name, profile, config)` — attributes `name`, `model`; `stream_chat` (raises `BackendError` on non-200 / network failure / timeout); `startup_check()` (llamacpp only: GET `<base without /v1>/health`, 3 s timeout, logs only); `close()`. One lazily-created `aiohttp.ClientSession` per instance.

- [ ] **Step 1: Write the failing test** `afterbuild/tests/test_openai_compat_http.py`:

```python
import asyncio

import pytest
from aiohttp import web

from backends.base import BackendError, ChatMessage, GenerationParams
from backends.openai_compat import OpenAICompatBackend

MESSAGES = [ChatMessage(role="user", content="hello")]
PARAMS = GenerationParams(max_tokens=20, temperature=0.0, top_p=1.0)


async def start_test_server(handler):
    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    app.router.add_get("/health", lambda request: web.Response(text="ok"))
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
    runner, port = await start_test_server(lambda request: web.Response(text="{}"))
    backend = make_backend(port)
    session = backend._get_session()
    await backend.close()
    assert session.closed
    await runner.cleanup()
```

- [ ] **Step 2: Run it, expect failure**

Run: `.venv\Scripts\python.exe -m pytest afterbuild/tests/test_openai_compat_http.py -v`
Expected: FAIL — `ImportError: cannot import name 'OpenAICompatBackend'`

- [ ] **Step 3: Append to** `afterbuild/server/backends/openai_compat.py`:

```python
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
```

Also add `import asyncio` to the file's imports (top, next to `json`).

- [ ] **Step 4: Run the test, expect pass**

Run: `.venv\Scripts\python.exe -m pytest afterbuild/tests/test_openai_compat_http.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Run all backend tests, expect pass**

Run: `.venv\Scripts\python.exe -m pytest afterbuild/tests -v`
Expected: PASS (all tests so far)

- [ ] **Step 6: Commit**

```bash
git add afterbuild/server/backends/openai_compat.py afterbuild/tests/test_openai_compat_http.py
git commit -m "feat(afterbuild): add aiohttp SSE streaming backend with health check"
```

---

### Task 6: Backend factory and config defaults

**Files:**
- Modify: `afterbuild/server/backends/__init__.py`
- Create: `afterbuild/tests/test_backend_factory.py`

**Interfaces:**
- Consumes: `EchoBackend` (Task 2), `OpenAICompatBackend` (Task 5).
- Produces: `create_backend(config: dict) -> Backend`; `DEFAULT_PROVIDERS` dict; re-exports `Backend`, `BackendError`, `ChatMessage`, `GenerationParams`, `EchoBackend`, `OpenAICompatBackend`. `create_backend` reads `config["provider"]` (default `"llamacpp"`), returns `EchoBackend()` for the reserved name `"echo"`, merges `config.get("providers", {})` over `DEFAULT_PROVIDERS`, and raises `BackendError` for an unknown provider.

- [ ] **Step 1: Write the failing test** `afterbuild/tests/test_backend_factory.py`:

```python
import pytest

from backends import DEFAULT_PROVIDERS, EchoBackend, OpenAICompatBackend, create_backend
from backends.base import BackendError


def test_echo_is_reserved_regardless_of_profiles():
    backend = create_backend({"provider": "echo", "providers": {}})
    assert isinstance(backend, EchoBackend)


def test_old_config_without_provider_key_defaults_to_llamacpp():
    backend = create_backend({"model_file": "old.gguf", "max_tokens": 100})
    assert isinstance(backend, OpenAICompatBackend)
    assert backend.name == "llamacpp"
    assert backend.profile["base_url"] == DEFAULT_PROVIDERS["llamacpp"]["base_url"]


def test_user_profiles_override_defaults_per_provider():
    config = {
        "provider": "deepseek",
        "providers": {"deepseek": {"base_url": "https://example.test/v1", "model": "d"}},
    }
    backend = create_backend(config)
    assert backend.profile["base_url"] == "https://example.test/v1"
    assert backend.profile["model"] == "d"


def test_unknown_provider_raises():
    with pytest.raises(BackendError, match="nope"):
        create_backend({"provider": "nope"})
```

- [ ] **Step 2: Run it, expect failure**

Run: `.venv\Scripts\python.exe -m pytest afterbuild/tests/test_backend_factory.py -v`
Expected: FAIL — `ImportError: cannot import name 'DEFAULT_PROVIDERS'`

- [ ] **Step 3: Replace** `afterbuild/server/backends/__init__.py` with:

```python
"""Inference backends for the NPC server."""
from __future__ import annotations

from typing import Any, Dict

from .base import Backend, BackendError, ChatMessage, GenerationParams
from .echo_backend import EchoBackend
from .openai_compat import OpenAICompatBackend

DEFAULT_PROVIDERS: Dict[str, Dict[str, Any]] = {
    "llamacpp": {
        "base_url": "http://127.0.0.1:8080/v1",
        "model": "llama-local",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "meta-llama/llama-3.2-3b-instruct",
        "api_key_env": "OPENROUTER_API_KEY",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "api_key_env": "DEEPSEEK_API_KEY",
    },
}


def create_backend(config: Dict[str, Any]) -> Backend:
    provider = config.get("provider", "llamacpp")
    if provider == "echo":
        return EchoBackend()
    profiles = {**DEFAULT_PROVIDERS, **config.get("providers", {})}
    profile = profiles.get(provider)
    if profile is None:
        raise BackendError(f"No profile configured for provider '{provider}'")
    return OpenAICompatBackend(name=provider, profile=profile, config=config)


__all__ = [
    "Backend",
    "BackendError",
    "ChatMessage",
    "GenerationParams",
    "EchoBackend",
    "OpenAICompatBackend",
    "create_backend",
    "DEFAULT_PROVIDERS",
]
```

- [ ] **Step 4: Run the test, expect pass**

Run: `.venv\Scripts\python.exe -m pytest afterbuild/tests/test_backend_factory.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add afterbuild/server/backends/__init__.py afterbuild/tests/test_backend_factory.py
git commit -m "feat(afterbuild): add backend factory with provider defaults"
```

---

### Task 7: Dependencies and server config

**Files:**
- Modify: `requirements.txt`
- Modify: `afterbuild/server/config.json`
- Create: `afterbuild/pytest.ini`

**Interfaces:**
- Consumes: nothing (infrastructure).
- Produces: installable dependency set; the shipped provider config consumed by Task 8's `load_config` defaults.

- [ ] **Step 1: Discover installed versions to pin**

Run: `.venv\Scripts\python.exe -m pip show websockets nltk pytest-asyncio aiohttp | findstr /R "^Name ^Version"`
Expected: Name/Version pairs (e.g. `websockets` / `15.x.y`). Record them for Step 2.

- [ ] **Step 2: Edit `requirements.txt`**

Remove the lines `gpt4all==2.8.2` and `llama_cpp_python==0.3.16`. Add, keeping the file's alphabetical-ish order and pinning the versions from Step 1:

```
nltk==<version from Step 1>
pytest-asyncio==<version from Step 1>
websockets==<version from Step 1>
```

Leave the existing `aiohttp==3.12.15`, `pytest==8.4.1`, and the top `--extra-index-url` line untouched.

- [ ] **Step 3: Replace** `afterbuild/server/config.json` with:

```json
{
  "provider": "llamacpp",
  "providers": {
    "llamacpp": {
      "base_url": "http://127.0.0.1:8080/v1",
      "model": "llama-local"
    },
    "openrouter": {
      "base_url": "https://openrouter.ai/api/v1",
      "model": "meta-llama/llama-3.2-3b-instruct",
      "api_key_env": "OPENROUTER_API_KEY"
    },
    "deepseek": {
      "base_url": "https://api.deepseek.com/v1",
      "model": "deepseek-chat",
      "api_key_env": "DEEPSEEK_API_KEY"
    }
  },
  "max_tokens": 150,
  "temperature": 0.7,
  "top_k": 40,
  "top_p": 0.9,
  "repeat_penalty": 1.18,
  "repeat_last_n": 64,
  "max_memory_entries": 20,
  "active_context_size": 7,
  "request_timeout_s": 60,
  "websocket_port": 9999
}
```

- [ ] **Step 4: Create `afterbuild/pytest.ini`:**

```ini
[pytest]
asyncio_mode = auto
testpaths = tests
```

- [ ] **Step 5: Verify** dependencies and config parse

Run: `.venv\Scripts\python.exe -c "import aiohttp, websockets, nltk, pytest_asyncio, json; json.load(open('afterbuild/server/config.json', encoding='utf-8')); print('deps+config OK')"`
Expected: `deps+config OK`
Run: `cd afterbuild && ..\.venv\Scripts\python.exe -m pytest tests -q`
Expected: all tests PASS (asyncio_mode=auto applied)

- [ ] **Step 6: Commit**

```bash
git add requirements.txt afterbuild/server/config.json afterbuild/pytest.ini
git commit -m "chore(afterbuild): add provider config, drop GPT4All deps, pin websockets/nltk"
```

---

### Task 8: Refactor dialogue_server onto the backend layer

**Files:**
- Modify: `afterbuild/server/dialogue_server.py`
- Create: `afterbuild/tools/ws_smoke_test.py`
- Create: `afterbuild/tests/test_dialogue_messages.py`

**Interfaces:**
- Consumes: everything from Tasks 1–7.
- Produces: `DialogueServer(config_path="config.json", decision_config_path="decision_config.json", memory_dir="../npc_memories", decision_log_dir="../decision_logs")`; methods `build_dialogue_messages(npc_name, from_speaker, user_message) -> list[ChatMessage]`, `dialogue_params() -> GenerationParams`, `decision_params() -> GenerationParams`, `async make_simple_decision(npc_name, context) -> dict`; `ws_smoke_test.py` CLI (`--port` external mode, default in-process echo mode, exit code 0 = pass).

- [ ] **Step 1: Write the failing unit tests** `afterbuild/tests/test_dialogue_messages.py`:

```python
import json
from pathlib import Path

import pytest

import dialogue_server
from backends import ChatMessage
from dialogue_server import DialogueServer


def make_server(tmp_path: Path) -> DialogueServer:
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "provider": "echo",
        "max_tokens": 50,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.1,
        "repeat_last_n": 32,
        "active_context_size": 2,
    }), encoding="utf-8")
    decision = tmp_path / "decision_config.json"
    decision.write_text(json.dumps({
        "valid_actions": ["idle"],
        "npc_specific_actions": {"default": ["idle"]},
        "npc_specific_targets": {"default": ["self"]},
        "npc_prompts": {"default": {"template": "{npc} {actions} {targets} {context}"}},
        "system_prompts": {"strict": "decision bot"},
        "generation_params": {"max_tokens": 15, "temperature": 0.3, "top_p": 0.5, "top_k": 10},
    }), encoding="utf-8")
    return DialogueServer(
        config_path=str(config),
        decision_config_path=str(decision),
        memory_dir=str(tmp_path / "memories"),
        decision_log_dir=str(tmp_path / "logs"),
    )


def memory(user: str, reply: str) -> dict:
    return {"user_input": user, "npc_response": reply, "keywords": ["chat"],
            "speaker": "user", "importance": 3.0}


def test_user_dialogue_includes_history_and_new_message(tmp_path):
    server = make_server(tmp_path)
    server.memory_cache["Leonardo"] = [
        memory("one", "reply one"),
        memory("two", "reply two"),
        memory("three", "reply three"),
    ]
    messages = server.build_dialogue_messages("Leonardo", "user", "hello now")
    roles = [m.role for m in messages]
    assert roles[0] == "system"
    assert roles[-1] == "user" and messages[-1].content == "hello now"
    # active_context_size=2 -> only the last two memories become turns
    assert "one" not in [m.content for m in messages]
    assert messages[1].content == "two" and messages[2].content == "reply two"
    assert messages[3].content == "three" and messages[4].content == "reply three"


def test_memories_with_empty_sides_are_skipped(tmp_path):
    server = make_server(tmp_path)
    server.memory_cache["Leonardo"] = [
        {"user_input": "", "npc_response": "orphan", "keywords": []},
        memory("kept", "kept reply"),
    ]
    messages = server.build_dialogue_messages("Leonardo", "user", "next")
    # Memory snippets may still appear in the system prompt (pre-existing
    # behavior); the *turn history* must skip the empty-sided entry.
    turn_contents = [m.content for m in messages[1:]]
    assert "orphan" not in turn_contents
    assert "kept" in turn_contents and "kept reply" in turn_contents


def test_empty_memory_cache_still_produces_system_and_user(tmp_path):
    server = make_server(tmp_path)
    server.memory_cache["Leonardo"] = []
    messages = server.build_dialogue_messages("Leonardo", "user", "hi")
    assert [m.role for m in messages] == ["system", "user"]


def test_npc_to_npc_has_no_history(tmp_path):
    server = make_server(tmp_path)
    server.memory_cache["Einstein"] = [memory("old", "old reply")]
    messages = server.build_dialogue_messages("Einstein", "Shakespeare", "speak, thinker")
    assert [m.role for m in messages] == ["system", "user"]
    assert "Shakespeare" in messages[0].content
    assert messages[1].content == "speak, thinker"


def test_system_generation_uses_message_as_system_prompt(tmp_path):
    server = make_server(tmp_path)
    messages = server.build_dialogue_messages("Socrates", "system", "You are Socrates.")
    assert messages[0] == ChatMessage(role="system", content="You are Socrates.")
    assert messages[1].content == "Start a conversation."


def test_dialogue_and_decision_params(tmp_path):
    server = make_server(tmp_path)
    dialogue = server.dialogue_params()
    assert (dialogue.max_tokens, dialogue.top_k, dialogue.repeat_penalty) == (50, 40, 1.1)
    decision = server.decision_params()
    assert (decision.max_tokens, decision.temperature, decision.top_k) == (15, 0.3, 10)


def test_save_memory_records_provider_model(tmp_path):
    server = make_server(tmp_path)
    server.save_memory("Leonardo", "q", "a", 0.5, from_speaker="user")
    saved = json.loads((tmp_path / "memories" / "Leonardo.json").read_text(encoding="utf-8"))
    assert saved[-1]["metadata"]["model"] == "echo:echo"
```

- [ ] **Step 2: Run them, expect failure**

Run: `.venv\Scripts\python.exe -m pytest afterbuild/tests/test_dialogue_messages.py -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'memory_dir'` (or `AttributeError: build_dialogue_messages`)

- [ ] **Step 3: Refactor `dialogue_server.py`**

Make these exact changes:

a. Replace the imports:

```python
from gpt4all import GPT4All
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
```

with:

```python
from backends import (
    Backend,
    BackendError,
    ChatMessage,
    GenerationParams,
    create_backend,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import nltk

    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords as _nltk_stopwords

    stop_words = set(_nltk_stopwords.words('english'))
except Exception as _exc:  # offline first run etc. — keep serving with reduced list
    logger.warning("nltk stopwords unavailable (%s); continuing with reduced stopword list", _exc)
    stop_words = set()
```

and delete the now-duplicate `logging.basicConfig(...)` / `logger = logging.getLogger(__name__)` lines that followed the old block.

b. Replace the `__init__` signature and the model/session/lock block:

```python
    def __init__(self, config_path: str = "config.json", decision_config_path: str = "decision_config.json"):
        """Initialize with configuration"""
        self.config = self.load_config(config_path)
        self.decision_config = self.load_decision_config(decision_config_path)
        self.model = None
        self.npc_sessions = {}  # Store chat sessions for each NPC
        self.decision_sessions = {}  # Store decision sessions (separate from dialogue)
        
        # Step 1: Basic lock for model access
        self.model_lock = asyncio.Lock()
        
        # Memory directory
        self.memory_dir = Path("../npc_memories")
        self.memory_dir.mkdir(exist_ok=True)
        
        # Decision log directory
        self.decision_log_dir = Path("../decision_logs")
        self.decision_log_dir.mkdir(exist_ok=True)
```

with:

```python
    def __init__(
        self,
        config_path: str = "config.json",
        decision_config_path: str = "decision_config.json",
        memory_dir: str = "../npc_memories",
        decision_log_dir: str = "../decision_logs",
    ):
        """Initialize with configuration"""
        self.config = self.load_config(config_path)
        self.decision_config = self.load_decision_config(decision_config_path)
        self.backend = create_backend(self.config)
        self.active_context_size = int(self.config.get("active_context_size", 7))

        # Memory directory
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)

        # Decision log directory
        self.decision_log_dir = Path(decision_log_dir)
        self.decision_log_dir.mkdir(exist_ok=True)
```

c. In `load_config`, replace the `else:` default dict with the new schema (mirroring `afterbuild/server/config.json` from Task 7, minus comments):

```python
            default_config = {
                "provider": "llamacpp",
                "max_tokens": 150,
                "temperature": 0.7,
                "top_k": 40,
                "top_p": 0.9,
                "repeat_penalty": 1.18,
                "repeat_last_n": 64,
                "max_memory_entries": 20,
                "active_context_size": 7,
                "request_timeout_s": 60,
                "websocket_port": 9999,
            }
```

d. Delete `load_model()` entirely and delete `get_or_create_session()` and `get_or_create_decision_session()`. Add in their place:

```python
    def build_dialogue_messages(self, npc_name: str, from_speaker: str, user_message: str) -> List[ChatMessage]:
        """Build the full stateless message list for one dialogue turn."""
        if from_speaker == "system":
            return [
                ChatMessage(role="system", content=user_message),
                ChatMessage(role="user", content="Start a conversation."),
            ]

        if from_speaker == "user":
            prompts = {
                "Leonardo": "You are Leonardo da Vinci, Renaissance genius bartender.\nSpeak with curiosity about art, science, and inventions.\nReply with ONE short sentence only.",
                "Einstein": "You are Albert Einstein, brilliant physicist contemplating in a bar.\nSpeak with gentle humor about relativity and the universe.\nReply with ONE short sentence only.",
                "Shakespeare": "You are William Shakespeare, the great playwright.\nSpeak dramatically and poetically, with theatrical flair.\nReply with ONE short sentence only.",
                "Socrates": "You are Socrates, the ancient philosopher as a wise dog.\nAsk thought-provoking questions with barks of wisdom.\nReply with ONE short sentence only."
            }
            system_prompt = prompts.get(npc_name, f"You are {npc_name}. {from_speaker} is talking to you. Reply with ONE short sentences only.")

            if self.memory_cache.get(npc_name):
                relevant_memories = self.retrieve_relevant_memories(
                    npc_name, from_speaker, speaker=from_speaker, k=5
                )
                if relevant_memories:
                    memory_text = "\n\nMost relevant memories:\n"
                    for memory in relevant_memories:
                        speaker_name = memory.get('speaker', 'User')
                        keywords = memory.get('keywords', [])
                        memory_text += (
                            f"- [{', '.join(keywords[:3])}] {speaker_name}: "
                            f"{memory.get('user_input', '')[:50]}...\n"
                        )
                        memory_text += (
                            f"  You: {memory.get('npc_response', '')[:50]}...\n"
                        )
                    system_prompt += memory_text

            messages = [ChatMessage(role="system", content=system_prompt)]
            for memory in self.memory_cache.get(npc_name, [])[-self.active_context_size:]:
                user_input = memory.get("user_input", "")
                response = memory.get("npc_response", "")
                if user_input and response:
                    messages.append(ChatMessage(role="user", content=user_input))
                    messages.append(ChatMessage(role="assistant", content=response))
            messages.append(ChatMessage(role="user", content=user_message))
            return messages

        # NPC-to-NPC: relationship-aware prompt, no history (matches previous fresh-session behavior)
        relationships = {
            ("Einstein", "Leonardo"): "Leonardo, the Renaissance master",
            ("Leonardo", "Einstein"): "Einstein, the modern genius",
            ("Shakespeare", "Leonardo"): "Leonardo, the artistic soul",
            ("Leonardo", "Shakespeare"): "Shakespeare, the wordsmith",
            ("Einstein", "Shakespeare"): "Shakespeare, the dramatic poet",
            ("Shakespeare", "Einstein"): "Einstein, the cosmic thinker",
            ("Socrates", "Leonardo"): "Leonardo, the polymath",
            ("Leonardo", "Socrates"): "Socrates, the wise hound",
            ("Socrates", "Einstein"): "Einstein, the truth seeker",
            ("Einstein", "Socrates"): "Socrates, the philosopher dog",
            ("Socrates", "Shakespeare"): "Shakespeare, the bard",
            ("Shakespeare", "Socrates"): "Socrates, the questioning canine"
        }
        speaker_desc = relationships.get((from_speaker, npc_name), from_speaker)
        prompts = {
            "Leonardo": f"You are Leonardo da Vinci. {speaker_desc} is talking to you.\nRespond with Renaissance curiosity and artistic insight.\nReply with ONE short sentence only.",
            "Einstein": f"You are Albert Einstein. {speaker_desc} is talking to you.\nRespond with scientific wonder and gentle humor.\nReply with ONE short sentence only.",
            "Shakespeare": f"You are William Shakespeare. {speaker_desc} is talking to you.\nRespond dramatically with poetic flair.\nReply with ONE short sentence only.",
            "Socrates": f"You are Socrates, a philosopher in dog form. {speaker_desc} is talking to you.\nRespond with wisdom or a thought-provoking question.\nReply with ONE short sentence only."
        }
        system_prompt = prompts.get(npc_name, f"You are {npc_name}. Reply with ONE short sentence only.")
        return [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_message),
        ]

    def dialogue_params(self) -> GenerationParams:
        return GenerationParams(
            max_tokens=self.config["max_tokens"],
            temperature=self.config["temperature"],
            top_p=self.config["top_p"],
            top_k=self.config.get("top_k"),
            repeat_penalty=self.config.get("repeat_penalty"),
            repeat_last_n=self.config.get("repeat_last_n"),
        )

    def decision_params(self) -> GenerationParams:
        gen = self.decision_config["generation_params"]
        return GenerationParams(
            max_tokens=gen["max_tokens"],
            temperature=gen["temperature"],
            top_p=gen["top_p"],
            top_k=gen.get("top_k"),
        )
```

e. In `save_memory`, change:

```python
                'model': self.config['model_file']
```

to:

```python
                'model': f"{self.backend.name}:{self.backend.model}"
```

f. Make `make_simple_decision` async and swap the model call. Replace this block:

```python
        # Record start time
        start_time = time.time()
        
        # Generate response using stateless chat session
        # Each request creates a new session context
        with self.model.chat_session(system_prompt=system_prompt) as session:
            raw_response = session.generate(
                prompt,
                max_tokens=gen_params["max_tokens"],
                temp=gen_params["temperature"],
                top_k=gen_params["top_k"],
                top_p=gen_params["top_p"],
                streaming=False
            )
        
        elapsed_time = time.time() - start_time
```

with:

```python
        # Record start time
        start_time = time.time()

        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=prompt),
        ]
        try:
            raw_response = await self.backend.chat(messages, self.decision_params())
        except BackendError as exc:
            logger.error(f"Decision backend error for {npc_name}: {exc}")
            raw_response = ""

        elapsed_time = time.time() - start_time
```

and change the `def make_simple_decision(self, npc_name: str, context: str) -> Dict:` line to `async def ...`. Note `gen_params` above it becomes unused — delete the `gen_params = level_config["generation_params"]` line.

g. In `handle_websocket`, the decision branch: replace

```python
                        # Use lock to prevent concurrent model access
                        async with self.model_lock:
                            logger.info(f"[LOCK] {npc_name} decision acquired lock")
                            try:
                                # Make decision - this uses the model
                                result = self.make_simple_decision(npc_name, context)
                                logger.info(f"[LOCK] {npc_name} decision completed")
                            except Exception as e:
                                logger.error(f"[LOCK] {npc_name} decision failed: {e}")
                                result = {"action": "idle", "target": "self", "raw": "error", "time": 0, "valid": False}
                            # Lock automatically released here after try/except
                        
                        logger.info(f"[LOCK] {npc_name} decision lock released")
```

with:

```python
                        try:
                            result = await self.make_simple_decision(npc_name, context)
                        except Exception as e:
                            logger.error(f"{npc_name} decision failed: {e}")
                            result = {"action": "idle", "target": "self", "raw": "error", "time": 0, "valid": False}
```

h. Replace the entire dialogue-generation block — from `# Use lock for entire dialogue generation` through `logger.info(f"[LOCK] {npc_name} dialogue lock released")` — with:

```python
                    start_time = time.time()
                    full_response = ""
                    try:
                        messages = self.build_dialogue_messages(npc_name, from_speaker, user_message)
                        logger.info(f"[{npc_name}] Sending {len(messages)} messages via {self.backend.name}")
                        async for token in self.backend.stream_chat(messages, self.dialogue_params()):
                            await websocket.send(json.dumps({
                                "type": "token",
                                "content": token,
                                "npc": npc_name
                            }))
                            full_response += token
                            await asyncio.sleep(0.02)  # pacing for the Godot client
                    except BackendError as exc:
                        logger.error(f"[{npc_name}] Backend error: {exc}")
                        await websocket.send(json.dumps({"type": "error", "content": str(exc)}))
                    except Exception as exc:
                        logger.error(f"[{npc_name}] Dialogue failed: {exc}")
                        await websocket.send(json.dumps({"type": "error", "content": str(exc)}))

                    final_text = full_response.strip() or "Sorry, I'm having trouble responding right now."
```

The existing frames after the old block stay, with `full_response.strip()` replaced by `final_text` in the `complete` frame and the `save_memory` call.

i. `cleanup()` — replace the whole session-closing body with:

```python
    def cleanup(self):
        """Nothing persistent to close; memories are written on each interaction."""
```

j. `start_server()` — replace

```python
        if not self.load_model():
            return
        
        port = self.config.get("websocket_port", 9999)
```

with

```python
        await self.backend.startup_check()

        port = self.config.get("websocket_port", 9999)
```

update the banner lines (`Model: ...` / `Device: ...`) to:

```python
        print(f"Provider: {self.backend.name} ({self.backend.model})")
```

and wrap the serve:

```python
        try:
            async with websockets.serve(self.handle_websocket, "127.0.0.1", port):
                await asyncio.Future()  # Run forever
        finally:
            await self.backend.close()
```

- [ ] **Step 4: Run the unit tests, expect pass**

Run: `.venv\Scripts\python.exe -m pytest afterbuild/tests/test_dialogue_messages.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Write `afterbuild/tools/ws_smoke_test.py`:**

```python
#!/usr/bin/env python3
"""WebSocket protocol smoke test.

Default: starts an in-process server with the echo backend and checks the
frozen frame protocol. --port P: checks an already-running server instead
(e.g. one backed by llama-server).

Exit code 0 = all checks passed.
"""
import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parent.parent / "server"
sys.path.insert(0, str(SERVER_DIR))

import websockets  # noqa: E402

NPCS = ["Leonardo", "Einstein", "Shakespeare", "Socrates"]
ECHO_PORT = 19999
TIMEOUT_S = 60


async def check_dialogue(ws, npc: str) -> None:
    await ws.send(json.dumps({"npc": npc, "message": f"{npc}|Hello"}))
    tokens = 0
    while True:
        frame = json.loads(await asyncio.wait_for(ws.recv(), timeout=TIMEOUT_S))
        if frame["type"] == "token":
            tokens += 1
        elif frame["type"] == "complete":
            assert tokens >= 1, f"{npc}: complete without tokens"
            print(f"PASS dialogue {npc}: {tokens} tokens")
            return
        elif frame["type"] == "error":
            raise AssertionError(f"{npc}: error frame: {frame['content']}")


async def check_decision(ws, npc: str) -> None:
    await ws.send(json.dumps({"type": "decision", "npc": npc, "context": "smoke test"}))
    frame = json.loads(await asyncio.wait_for(ws.recv(), timeout=TIMEOUT_S))
    assert frame["type"] == "decision_result", frame
    assert "action" in frame and "target" in frame, frame
    print(f"PASS decision {npc}: {frame['action']}|{frame['target']}")


async def check_client_disconnect(port: int) -> None:
    """Abandon a stream mid-way; the server must survive and serve the next client."""
    ws = await websockets.connect(f"ws://127.0.0.1:{port}")
    await ws.send(json.dumps({"npc": "Leonardo", "message": "Leonardo|Hello"}))
    await asyncio.wait_for(ws.recv(), timeout=TIMEOUT_S)
    await ws.close()  # abandon while the server is still streaming
    await asyncio.sleep(0.2)
    async with websockets.connect(f"ws://127.0.0.1:{port}") as next_ws:
        await check_dialogue(next_ws, "Einstein")
    print("PASS client disconnect survival")


async def run_checks(port: int) -> None:
    async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
        for npc in NPCS:
            await check_dialogue(ws, npc)
        await check_decision(ws, NPCS[0])
    await check_client_disconnect(port)
    print("SMOKE TEST PASS")


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=None,
                        help="check an already-running server on this port")
    args = parser.parse_args()

    if args.port:
        await run_checks(args.port)
        return 0

    from dialogue_server import DialogueServer

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "config.json").write_text(json.dumps({
            "provider": "echo", "websocket_port": ECHO_PORT,
            "max_tokens": 50, "temperature": 0.7, "top_p": 0.9,
        }), encoding="utf-8")
        server = DialogueServer(
            config_path=str(tmp_path / "config.json"),
            decision_config_path=str(SERVER_DIR / "decision_config.json"),
            memory_dir=str(tmp_path / "memories"),
            decision_log_dir=str(tmp_path / "logs"),
        )
        task = asyncio.create_task(server.start_server())
        await asyncio.sleep(0.5)  # let it bind
        try:
            await run_checks(ECHO_PORT)
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
```

- [ ] **Step 6: Run the smoke test, expect pass**

Run: `.venv\Scripts\python.exe afterbuild/tools/ws_smoke_test.py`
Expected: `PASS dialogue ...` ×4, `PASS decision Leonardo: ...`, `PASS client disconnect survival`, `SMOKE TEST PASS`, exit code 0.

- [ ] **Step 7: Run the full test suite**

Run: `.venv\Scripts\python.exe -m pytest afterbuild/tests -v`
Expected: all PASS

- [ ] **Step 8: Commit**

```bash
git add afterbuild/server/dialogue_server.py afterbuild/tools/ws_smoke_test.py afterbuild/tests/test_dialogue_messages.py
git commit -m "refactor(afterbuild): run dialogue server on the backend provider layer"
```

---

### Task 9: Local integration with real llama-server

**Files:**
- No code changes expected (fix forward if something breaks; commit fixes separately).

**Interfaces:**
- Consumes: everything so far, plus the user's `llama-server.exe` and GGUF model.
- Produces: verified end-to-end local inference with memory persistence.

- [ ] **Step 1: Locate llama-server**

Run: `Get-Command llama-server -ErrorAction SilentlyContinue | Select-Object Source`
If not on PATH, ask the user for the path to their `llama-server.exe` and the GGUF file (default expected: `models\llms\Llama-3.2-3B-Instruct-Q4_0.gguf`).

- [ ] **Step 2: Start llama-server in the background** (redirects need the log directory to exist first)

```powershell
New-Item -ItemType Directory -Force logs | Out-Null
Start-Process -FilePath <llama-server.exe> -ArgumentList '-m','<repo>\models\llms\Llama-3.2-3B-Instruct-Q4_0.gguf','-ngl','99','-c','8192','-np','4','--port','8080','--jinja' -RedirectStandardOutput logs\llama-server.out.log -RedirectStandardError logs\llama-server.err.log
```

- [ ] **Step 3: Health check**

Run: `curl http://127.0.0.1:8080/health`
Expected: `{"status":"ok"}`

- [ ] **Step 4: Start the NPC server in the background**

```powershell
Start-Process -FilePath "$PWD\.venv\Scripts\python.exe" -ArgumentList 'dialogue_server.py' -WorkingDirectory "$PWD\afterbuild\server" -RedirectStandardOutput "$PWD\logs\npc-server.out.log" -RedirectStandardError "$PWD\logs\npc-server.err.log"
```
Expected in the log: `llama-server health OK`, `Provider: llamacpp (llama-local)`, `Waiting for connections...`

- [ ] **Step 5: Back up memories, then run the smoke test against the live server**

```powershell
Copy-Item afterbuild\npc_memories afterbuild\npc_memories.backup -Recurse
.venv\Scripts\python.exe afterbuild\tools\ws_smoke_test.py --port 9999
```
Expected: `SMOKE TEST PASS`. Note first-token latency in the log (`[<npc>] Response in ...s`).

- [ ] **Step 6: Verify memory metadata**

Run: `.venv\Scripts\python.exe -c "import json; d=json.load(open('afterbuild/npc_memories/Leonardo.json', encoding='utf-8')); print(d[-1]['metadata'])"`
Expected: `'model': 'llamacpp:llama-local'` present.

- [ ] **Step 7: Concurrency check** — the smoke test's 4 dialogues run on one socket sequentially; additionally verify overlap with two concurrent clients (llama-server `-np 4`):

Run:
```powershell
.venv\Scripts\python.exe -c "import asyncio, json, websockets
async def one(npc):
    async with websockets.connect('ws://127.0.0.1:9999') as ws:
        await ws.send(json.dumps({'npc': npc, 'message': npc + '|Hello'}))
        while True:
            f = json.loads(await ws.recv())
            if f['type'] == 'complete':
                print(npc, 'done'); return
async def main():
    await asyncio.gather(one('Leonardo'), one('Einstein'))
asyncio.run(main())"
```
Expected: both lines print; neither blocks the other.

- [ ] **Step 8: Stop background processes**

`Stop-Process` the two PIDs from Steps 2 and 4. Restore memories if the run was only a test: `Remove-Item afterbuild\npc_memories -Recurse; Move-Item afterbuild\npc_memories.backup afterbuild\npc_memories`.

---

### Task 10: Godot autotest scene

**Files:**
- Create: `afterbuild/godot_project/scenes/dev/autotest.tscn`
- Create: `afterbuild/godot_project/scripts/dev/autotest.gd`
- Create: `afterbuild/godot_project/scripts/dev/autotest.gd.uid` (generated by Godot on first import; commit whatever Godot creates)

**Interfaces:**
- Consumes: `dialogue_client.gd`'s API (frozen): `interact_with_npc(npc_name: String, custom_message: String = "")`, signals `token_received(npc, token)` / `response_completed(npc, full)`, var `is_websocket_connected`.
- Produces: `AUTOTEST PASS`/`AUTOTEST FAIL` lines on stdout, exits with `get_tree().quit()`.

- [ ] **Step 1: Write `afterbuild/godot_project/scripts/dev/autotest.gd`:**

```gdscript
extends Node2D
## End-to-end test: drives one dialogue per NPC through the real client stack.
## Run: godot --path afterbuild/godot_project res://scenes/dev/autotest.tscn
## Never set as the project's main scene.

const NPCS := ["Leonardo", "Einstein", "Shakespeare", "Socrates"]
const PER_NPC_TIMEOUT_S := 60.0
const CONNECT_TIMEOUT_S := 10.0

var bar: Node = null
var current_npc := ""
var npc_index := 0
var token_count := 0
var elapsed := 0.0
var waiting := false
var connect_elapsed := 0.0
var failed := false


func _ready() -> void:
	bar = load("res://scenes/cozy_bar.tscn").instantiate()
	add_child(bar)
	print("AUTOTEST: waiting for websocket...")


func _process(delta: float) -> void:
	if failed:
		return
	if not waiting:
		connect_elapsed += delta
		if bar.is_websocket_connected:
			_start_next()
		elif connect_elapsed > CONNECT_TIMEOUT_S:
			_fail("no websocket after %ss" % CONNECT_TIMEOUT_S)
		return
	elapsed += delta
	if elapsed > PER_NPC_TIMEOUT_S:
		_fail("timeout for %s" % current_npc)


func _start_next() -> void:
	if npc_index >= NPCS.size():
		print("AUTOTEST ALL PASS")
		get_tree().quit(0)
		return
	if not bar.response_completed.is_connected(_on_response_completed):
		bar.response_completed.connect(_on_response_completed)
		bar.token_received.connect(_on_token_received)
	current_npc = NPCS[npc_index]
	npc_index += 1
	token_count = 0
	elapsed = 0.0
	waiting = true
	print("AUTOTEST: sending to ", current_npc)
	bar.interact_with_npc(current_npc, "Hello from autotest!")


func _on_token_received(npc: String, _token: String) -> void:
	if npc == current_npc:
		token_count += 1


func _on_response_completed(npc: String, full: String) -> void:
	if npc != current_npc:
		return
	waiting = false
	print("AUTOTEST PASS npc=%s tokens=%d ms=%d text=%s" % [npc, token_count, int(elapsed * 1000.0), full.substr(0, 80)])
	if token_count < 1:
		_fail("%s completed with no tokens" % npc)
		return
	_start_next()


func _fail(reason: String) -> void:
	failed = true
	print("AUTOTEST FAIL: ", reason)
	get_tree().quit(1)
```

- [ ] **Step 2: Write `afterbuild/godot_project/scenes/dev/autotest.tscn`:**

```
[gd_scene load_steps=2 format=3]

[ext_resource type="Script" path="res://scripts/dev/autotest.gd" id="1"]

[node name="Autotest" type="Node2D"]
script = ExtResource("1")
```

(No `uid://` values — Godot fills them in on import. Create the `scenes/dev/` and `scripts/dev/` directories.)

- [ ] **Step 3: Parse check**

Run: `& $env:GODOT_PATH --headless --path afterbuild\godot_project --quit`
Expected: no script errors. (If `GODOT_PATH` is unset, use the full path: `"c:\program files\godot\Godot.exe"`.)

- [ ] **Step 4: Run end-to-end** (llama-server + NPC server running from Task 9)

Run: `& $env:GODOT_PATH --headless --path afterbuild\godot_project res://scenes/dev/autotest.tscn`
Expected: `AUTOTEST PASS` ×4, `AUTOTEST ALL PASS`, process exits 0. Alternative: run the scene via the Godot MCP tools if registered (`run_project`/`get_debug_output` — check actual names).

- [ ] **Step 5: Commit**

```bash
git add afterbuild/godot_project/scenes/dev afterbuild/godot_project/scripts/dev
git commit -m "test(afterbuild): add Godot autotest scene for end-to-end dialogue"
```

---

### Task 11: Cloud regression (OpenRouter + DeepSeek)

**Files:**
- No code changes expected.

**Interfaces:**
- Consumes: provider config (Task 7), running NPC server flow (Task 9).
- Produces: verified identical framing on cloud providers.

- [ ] **Step 1: OpenRouter run**

```powershell
$env:OPENROUTER_API_KEY = "<key>"
# edit afterbuild/server/config.json: "provider": "openrouter"
# start the NPC server (Task 9 Step 4), then:
.venv\Scripts\python.exe afterbuild\tools\ws_smoke_test.py --port 9999
```
Expected: `SMOKE TEST PASS`. Confirm `metadata.model` == `openrouter:meta-llama/llama-3.2-3b-instruct` in a fresh memory entry. Watch for the Godot-side 10 s dialogue timeout in the log timings — if first-token latency exceeds it for the autotest, note it to the user rather than changing the client.

- [ ] **Step 2: DeepSeek run**

Repeat Step 1 with `$env:DEEPSEEK_API_KEY` and `"provider": "deepseek"`; expect `deepseek:deepseek-chat` in metadata.

- [ ] **Step 3: Restore config** to `"provider": "llamacpp"` and stop processes.

---

### Task 12: Documentation

**Files:**
- Modify: `afterbuild/README.md`

**Interfaces:**
- Consumes: all previous tasks.
- Produces: reader-facing setup instructions.

- [ ] **Step 1: Rewrite the stale sections of `afterbuild/README.md`** — replace GPT4All references:
  - Requirements: `pip install -r requirements.txt` (aiohttp/websockets/nltk), plus running `llama-server` separately.
  - New "Inference providers" section: the three profile table, `provider` config key, env vars, and the exact `llama-server` launch command from CLAUDE.md §3.
  - Quick start: start llama-server → `cd afterbuild/server && python dialogue_server.py`.
  - Testing: `python tools/ws_smoke_test.py` (echo) and `--port 9999` against a live server; pytest command.
- [ ] **Step 2: Verify no GPT4All mentions remain in afterbuild/**

Run: `Get-ChildItem afterbuild -Recurse -File | Select-String -Pattern gpt4all -SimpleMatch -List | Select-Object Path`
Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add afterbuild/README.md
git commit -m "docs(afterbuild): document inference providers and llama-server setup"
```

---

## Final verification (after all tasks)

- [ ] `.venv\Scripts\python.exe -m pytest afterbuild/tests -v` — all pass.
- [ ] `.venv\Scripts\python.exe afterbuild/tools/ws_smoke_test.py` — `SMOKE TEST PASS`.
- [ ] `git status` — clean except CLAUDE.md (ignored) and the user's pre-existing `requirements.txt` torch line if still uncommitted.
- [ ] Push the branch: `git push` (branch `provider-layer` is already tracking origin).
