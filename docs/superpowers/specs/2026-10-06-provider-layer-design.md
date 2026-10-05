# Provider Layer Design — afterbuild NPC Server

Date: 2026-10-06
Status: Draft for review
Target: `afterbuild/` only (`finalbuild/` is superseded legacy; GPT4All files already deleted 2026-10-06)

## 1. Context and goal

`afterbuild/server/dialogue_server.py` currently runs all inference through the
GPT4All Python library (`GPT4All` + `chat_session`), which keeps conversation
state inside the model object and blocks the event loop during generation.

Goal: replace GPT4All with a provider layer supporting three interchangeable
backends — all of which expose the **OpenAI Chat Completions API**:

| Provider | Base URL | Auth | Model set by |
|---|---|---|---|
| `llamacpp` | `http://127.0.0.1:8080/v1` | none | `llama-server` launch command |
| `openrouter` | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` env | config |
| `deepseek` | `https://api.deepseek.com/v1` | `DEEPSEEK_API_KEY` env | config |

One provider serves everything (dialogue + decisions). No per-role override.

Constraints (from CLAUDE.md):

- The Godot WebSocket protocol is untouchable (frames, 20 ms token pacing,
  `complete`/`error` always sent).
- No llama.cpp Python dependency — llama.cpp is consumed only as the external
  `llama-server` binary.
- No GPT4All fallback.
- New config keys must have defaults so an old config file still works.
- Do not change the memory JSON schema (Godot reads it directly).

## 2. Non-goals

- No Godot-side changes (the autotest scene is separate work, see CLAUDE.md §5).
- No per-role / per-NPC provider selection.
- No runtime provider switching (config + restart).
- No changes to decision parsing, memory keyword logic, or prompt content.
- No retry/backoff layer (a failed request is reported, not retried).

## 3. Architecture

New package `afterbuild/server/backends/`:

```
backends/
  __init__.py        # create_backend(config) factory + re-exports
  base.py            # ChatMessage, GenerationParams, Backend ABC, BackendError
  openai_compat.py   # single HTTP/SSE client for all three providers
  echo_backend.py    # no-model backend for protocol tests
```

### 3.1 Interface (`base.py`)

```python
@dataclass(frozen=True)
class ChatMessage:
    role: str            # "system" | "user" | "assistant"
    content: str

@dataclass(frozen=True)
class GenerationParams:
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int | None = None            # llama.cpp-only; ignored elsewhere
    repeat_penalty: float | None = None # llama.cpp-only
    repeat_last_n: int | None = None    # llama.cpp-only

class BackendError(Exception):
    """Raised for any provider failure: transport, HTTP status, malformed SSE, missing key."""

class Backend(abc.ABC):
    name: str    # provider key, e.g. "llamacpp"
    model: str   # model id from config

    @abc.abstractmethod
    def stream_chat(self, messages: list[ChatMessage],
                    params: GenerationParams) -> AsyncIterator[str]:
        """Yield content deltas. Raises BackendError on failure."""

    async def chat(self, messages, params) -> str:
        """Accumulate stream_chat() into one string. Never overridden by subclasses."""

    async def startup_check(self) -> None:
        """Optional connectivity probe; logs, never raises."""

    async def close(self) -> None: ...
```

`stream_chat` is an async generator (not `async def ... -> AsyncIterator` with
inner generator complexity — implement as `async def` + `yield` directly).
`chat()` lives on the base class so there is exactly one accumulation path;
the decision code path calls `chat()`.

### 3.2 Factory (`__init__.py`)

```python
def create_backend(config: dict) -> Backend:
    provider = config.get("provider", "llamacpp")
    profiles = config.get("providers", {})
    if provider == "echo":
        return EchoBackend()
    profile = profiles.get(provider)
    if profile is None:
        raise BackendError(f"No profile configured for provider '{provider}'")
    return OpenAICompatBackend(name=provider, profile=profile, config=config)
```

`echo` is a reserved provider name, usable regardless of profiles.

### 3.3 OpenAI-compatible client (`openai_compat.py`)

- Uses `aiohttp.ClientSession` (already in requirements).
- Request: `POST {base_url}/chat/completions` with
  `Accept: text/event-stream`, JSON body:

  ```json
  {
    "model": "<profile.model>",
    "messages": [{"role": ..., "content": ...}, ...],
    "stream": true,
    "max_tokens": ..., "temperature": ..., "top_p": ...
  }
  ```

  Plus, **only when `name == "llamacpp"`** (llama-server extensions; cloud APIs
  reject unknown fields):

  ```json
  { "cache_prompt": true, "top_k": ..., "repeat_penalty": ..., "repeat_last_n": ... }
  ```

  Non-`None` values only. `id_slot` pinning is explicitly out of scope for v1.

- Headers: `Authorization: Bearer <key>` when `profile.api_key_env` is set.
  Key read from `os.environ[api_key_env]` at request time (not startup), so
  rotating the env var doesn't require a restart. Missing key → `BackendError`
  naming the env var. No keys are ever logged.
- SSE parsing: iterate response lines; ignore blank lines and `:` comments;
  strip `data: ` prefix; stop on `[DONE]`; JSON-decode each chunk and yield
  `choices[0].delta.content` when present (empty deltas are skipped, not errors).
  A chunk without `choices` is skipped. Malformed JSON → `BackendError`.
- Streaming body text is decoded with `errors="replace"` (llama-server emits
  plain UTF-8; replacement avoids hard crashes on a split multi-byte char in a
  malformed stream — normal splits are handled by aiohttp's incremental decoder).
- Timeouts: `aiohttp.ClientTimeout(total=request_timeout_s from config, default
  60 s; connect=10 s)`.
- Non-200 response → `BackendError` with status + first 200 chars of body.
- Network errors (`aiohttp.ClientError`, `asyncio.TimeoutError`) → `BackendError`.
- `startup_check()`: only for `name == "llamacpp"` — `GET <base_url without
  /v1>/health`, 3 s timeout, log ok/warn. Never raises. Cloud providers: no-op
  (don't pay for a startup probe).
- One `ClientSession` per backend instance, created lazily and closed in
  `close()`.

### 3.4 Echo backend (`echo_backend.py`)

Yields the words of `"Echo: " + <last user message content>` as individual
deltas. No delay, no network. Used by the protocol smoke test and unit tests.
Decision requests through echo accumulate to `Echo: <prompt>` → the decision
parser maps it to `idle|self` with `valid: false`, which is the expected
fallback path being exercised.

## 4. Config schema

`afterbuild/server/config.json` gains (existing keys unchanged):

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
  "active_context_size": 7,
  "request_timeout_s": 60
}
```

Defaults when keys are absent (old config files keep working):

- `provider` → `"llamacpp"`
- `providers` → the three profiles above with these exact values
- `active_context_size` → `7`
- `request_timeout_s` → `60`

Legacy GPT4All keys (`model_file`, `model_path`, `device`) are ignored by the
new code; they are removed from the shipped `config.json` but tolerated if an
old file still carries them.

The `llamacpp` profile's `model` is a label only — classic `llama-server`
ignores the request's model field when a single model is loaded; the value
appears in memory metadata and logs.

## 5. dialogue_server.py changes

Kept unchanged: `canonicalize_npc_name`, memory cache/loading/saving,
keyword extraction, decision parsing (`process_decision_output_with_target`),
decision logging, WS frame shapes, 20 ms inter-token sleep, `clean_response`.

### 5.1 Removed

- `GPT4All` import, `load_model()`, `self.model`, all `chat_session` usage.
- `get_or_create_session`, `get_or_create_decision_session`, `self.npc_sessions`,
  `self.decision_sessions`, and the session-closing loops in `cleanup()`
  (there is no in-process state to close).
- `self.model_lock` — the HTTP client is concurrency-safe and llama-server
  runs `-np 4` slots; concurrent NPC requests now genuinely overlap. Locked
  serialization was an artifact of the in-process model.

### 5.2 Added / changed

- `self.backend = create_backend(self.config)` in `__init__` (no model load).
- `start_server()` calls `await self.backend.startup_check()` before serving;
  the server starts regardless of the result (llama-server may be started
  after the NPC server), and wraps the serve loop in `try/finally` calling
  `await self.backend.close()`. `cleanup()` keeps only the memory-cache save.
- **Message building** — replaces `get_or_create_session()` with a pure
  function `build_dialogue_messages(npc_name, from_speaker, user_message) ->
  list[ChatMessage]`:
  - `from_speaker == "system"`: `[system: <user_message>, user: "Start a
    conversation."]` (preserves current behavior; the message *is* the system
    prompt).
  - `from_speaker == "user"`: system = existing per-NPC prompt **plus the
    existing "Most relevant memories" block** (verbatim from today's
    `get_or_create_session`), then the last `active_context_size` turns from
    `memory_cache[npc]` as `user: memory.user_input` / `assistant:
    memory.npc_response`, then the new user message.
  - `from_speaker` is another NPC: system = existing relationship prompt
    (verbatim), then just the new message — no history, matching today's
    fresh-session behavior.
  - Memory turns where either `user_input` or `npc_response` is empty are
    skipped.
- **Dialogue generation** becomes:

  ```python
  async for token in self.backend.stream_chat(messages, params):
      await websocket.send(json.dumps({"type": "token", "content": token, "npc": npc_name}))
      full_response += token
      await asyncio.sleep(0.02)
  ```

  where `params = GenerationParams(max_tokens=..., temperature=..., top_p=...,
  top_k=..., repeat_penalty=..., repeat_last_n=...)` built from the existing
  top-level config keys.
- **Decision generation** becomes `raw_response = await self.backend.chat(
  messages, decision_params)` inside `make_simple_decision` (now `async`);
  messages = `[system: decision system_prompts.strict, user: <built prompt>]`.
  Decision params come from `decision_config["generation_params"]` (its
  `top_k` maps to `GenerationParams.top_k`).
- `save_memory()` metadata: `"model": f"{self.backend.name}:{self.backend.model}"`
  (replaces the `"... (GPT4All)"` string).
- Startup banner prints provider + model instead of model file/device.

### 5.3 Error handling

Per request, in the WS handler:

- Any `BackendError` (or unexpected exception) during dialogue generation:
  log it; send `{"type": "error", "content": <human-readable reason>}`; then
  send `{"type": "complete", "content": <accumulated partial OR the existing
  fallback "Sorry, I'm having trouble responding right now.">, "npc": ...}`.
  If tokens were already streamed, `complete` carries the partial text (the
  client shows what arrived); otherwise the fallback sentence.
- Decision failure: log; respond with the existing error-shaped
  `decision_result` (`action: "idle", target: "self", valid: false,
  debug.raw: "error"`). Never leave the frame missing.
- `complete`/`decision_result` are always sent — the no-stuck-client contract.

## 6. Requirements

`requirements.txt`: remove `gpt4all==2.8.2` and `llama_cpp_python==0.3.16`;
add `websockets` (pin to the version active in `.venv` at implementation time)
and `nltk` (imported by `dialogue_server.py` but currently unpinned/missing).
`aiohttp` is already present.

Note: `dialogue_server.py` calls `nltk.download('stopwords')` at import; keep
it but wrap in `try/except` with a clear log line, so first run offline
doesn't crash the server.

## 7. Testing

1. **Unit (pytest, `afterbuild/tests/test_openai_compat.py`)**, mocking the
   aiohttp response with canned SSE byte streams:
   - normal stream → deltas in order, `[DONE]` terminates;
   - empty `delta.content` / chunk without `choices` → skipped;
   - malformed JSON chunk → `BackendError`;
   - `error` mid-stream → `BackendError` after prior deltas;
   - non-200 → `BackendError` with status and body snippet;
   - missing `api_key_env` var → `BackendError` naming the variable;
   - llama.cpp extras present iff provider is `llamacpp`;
   - `chat()` accumulates `stream_chat()` output.
2. **Protocol smoke (`afterbuild/tools/ws_smoke_test.py`)** against the echo
   backend: for each of Leonardo/Einstein/Shakespeare/Socrates send
   `{"npc": X, "message": "X|Hello"}` and assert ≥1 `token` then one
   `complete`; send a `decision` message and assert one `decision_result`.
3. **Integration** (manual, per CLAUDE.md §6 steps 2–4): llama-server alone →
   NPC server + llama-server (memory metadata + two concurrent sockets) →
   Godot autotest scene.
4. **Cloud regression** (CLAUDE.md §6 step 5): `openrouter`, then `deepseek`.

## 8. Risks and notes

- Cloud providers add first-token latency; the Godot client's 10 s dialogue
  timeout is the hard bound. The 20 ms inter-token sleep stays (client
  contract); if cloud streams make it a problem, lower it deliberately in a
  separate change.
- llama-server's request `model` field is ignored in single-model mode — this
  is expected, not a bug; do not "fix" it by validating.
- SSE chunk shapes differ subtly between providers (e.g. DeepSeek sends
  `usage` in the final chunk; OpenRouter may include `: OPENROUTER PROCESSING`
  comments). The parser must ignore anything without
  `choices[0].delta.content`.
- `active_context_size` history + the memory-snippets system block overlap;
  that is intentional for now (continuity + relevance). Reducing one is a
  later tuning question, not a v1 requirement.

## 9. File inventory

New:

- `afterbuild/server/backends/__init__.py`
- `afterbuild/server/backends/base.py`
- `afterbuild/server/backends/openai_compat.py`
- `afterbuild/server/backends/echo_backend.py`
- `afterbuild/tests/test_openai_compat.py`
- `afterbuild/tools/ws_smoke_test.py`

Modified:

- `afterbuild/server/dialogue_server.py`
- `afterbuild/server/config.json`
- `requirements.txt`
- `afterbuild/README.md` (provider setup, after the code works)
