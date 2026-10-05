# Afterbuild - Simplified Dialogue System

A clean, unified dialogue system for NPCs with a pluggable inference layer,
designed for progressive development.

## Architecture

### Simplified Design
```
Godot Client ←→ WebSocket ←→ Unified Server ←→ Inference Provider
                                                ├── llama-server (local, llama.cpp)
                                                ├── OpenRouter (cloud)
                                                └── DeepSeek (cloud)
```

**Key improvements**: the server merges the former client and server scripts
into a single unified server (no intermediate socket layer), and all inference
now runs through one OpenAI-compatible HTTP client — the old in-process model
backend is gone.

## Project Structure

```
afterbuild/
├── server/
│   ├── dialogue_server.py       # WebSocket + memory + decision parsing
│   ├── backends/                # provider layer
│   │   ├── base.py              # Backend interface + message/param types
│   │   ├── openai_compat.py     # SSE client for all three providers
│   │   └── echo_backend.py      # no-model backend for protocol tests
│   ├── config.json              # provider + sampling configuration
│   └── decision_config.json     # decision prompts/actions/targets
├── godot_project/
│   └── scripts/dialogue_client.gd
├── tools/ws_smoke_test.py       # WebSocket protocol smoke test
├── tests/                       # pytest suite
├── npc_memories/                # NPC conversation histories
└── README.md
```

## Core Features

### 1. Dialogue Server (`dialogue_server.py`)
- **Single Process**: no need for separate client/server processes
- **Direct WebSocket**: Godot connects directly, no intermediate layers
- **Stateless message building**: each request sends system prompt + recent
  memory turns + the new message (works identically on local and cloud)
- **Memory Management**: built-in conversation history
- **NPC Name Canonicalization**: prevents memory mixing (Leonardo/leonardo/davinci → Leonardo)
- **Streaming Support**: real-time token streaming for natural dialogue

### 2. Inference Providers

All three providers speak the OpenAI Chat Completions API; pick one with the
`provider` key in `server/config.json`:

| Provider | Base URL | Key (env var) | Model |
|---|---|---|---|
| `llamacpp` (local) | `http://127.0.0.1:8080/v1` | none | set when launching `llama-server` |
| `openrouter` | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` | any OpenRouter model id |
| `deepseek` | `https://api.deepseek.com/v1` | `DEEPSEEK_API_KEY` | `deepseek-chat` |

```json
{
  "provider": "llamacpp",
  "providers": {
    "llamacpp": { "base_url": "http://127.0.0.1:8080/v1", "model": "gemma-4-31B-it-heretic" },
    "openrouter": { "base_url": "https://openrouter.ai/api/v1", "model": "meta-llama/llama-3.2-3b-instruct", "api_key_env": "OPENROUTER_API_KEY" },
    "deepseek": { "base_url": "https://api.deepseek.com/v1", "model": "deepseek-chat", "api_key_env": "DEEPSEEK_API_KEY" }
  }
}
```

API keys are read from the environment at request time — never commit them.
`"provider": "echo"` is reserved for no-model protocol testing.

### 3. Godot Client (`dialogue_client.gd`)
- WebSocket connection to the unified server
- Speech bubble UI system
- Memory viewer interface
- Supports Leonardo, Einstein, Shakespeare, and Socrates NPCs

## Quick Start

### 1. Start llama-server (local provider)

```powershell
D:\ai\llamacpp\llama-server.exe `
  -m D:\ai\text\models\gemma-4-31B-it-heretic\gemma-4-31B-it-heretic.i1-Q4_K_M.gguf `
  -ngl 99 -c 8192 -np 4 --port 8080 --jinja --reasoning off
```

`--reasoning off` matters for thinking models: without it the model spends
`max_tokens` on hidden reasoning and the NPCs say nothing. `-np 4` gives one
slot per NPC. The model takes ~1 minute to load; `curl http://127.0.0.1:8080/health`
returns 503 until it is ready.

### 2. Start the Server

```bash
cd afterbuild/server
python dialogue_server.py
```

The server will:
- create the configured provider backend
- check llama-server's health (local provider only; the server still starts if it's down)
- start the WebSocket server on port 9999
- load existing NPC memories
- wait for Godot connections

### 3. Connect from Godot
The Godot client automatically connects to `ws://127.0.0.1:9999`

### 4. Interact with NPCs
- Left-click NPC: Send default greeting
- Right-click NPC: Custom message dialog

## Testing

```bash
# All unit tests (pytest + pytest-asyncio)
.venv/Scripts/python.exe -m pytest afterbuild/tests -v

# Protocol smoke test, no model needed (starts an in-process echo server)
.venv/Scripts/python.exe afterbuild/tools/ws_smoke_test.py

# Same checks against a running server (any provider), e.g. llama-server
.venv/Scripts/python.exe afterbuild/tools/ws_smoke_test.py --port 9999

# Godot end-to-end (needs the NPC server running first)
godot --headless --path afterbuild/godot_project res://scenes/dev/autotest.tscn
```

The autotest scene drives one dialogue per NPC through the real client stack and
prints `AUTOTEST PASS` / `AUTOTEST ALL PASS`. Never set it as the main scene.

## Memory System

Conversations are automatically saved to `npc_memories/`:
- `Leonardo.json`, `Einstein.json`, `Shakespeare.json`, `Socrates.json`

Each memory entry contains:
- Timestamp
- User input
- NPC response
- Response time
- `metadata.model` = `"<provider>:<model>"` (e.g. `llamacpp:gemma-4-31B-it-heretic`)

## Development Roadmap

### Phase 1: Core Dialogue ✅
- Basic conversation system
- Memory persistence
- Multi-NPC support

### Phase 2: Enhanced Memory (Future)
- Importance-based memory retention
- Long-term vs short-term memory
- Memory summarization

### Phase 3: Context Awareness (Future)
- Environment state integration
- Time-of-day awareness
- Mood and relationship tracking

### Phase 4: Advanced Features (Future)
- Emotion detection
- Dynamic personality adjustment
- Inter-NPC conversations

## Key Advantages

1. **Simplicity**: single Python process, no complex networking
2. **Portability**: same protocol against a local model or a cloud API
3. **Performance**: truly async streaming (no event-loop blocking)
4. **Maintainability**: one OpenAI-compatible client for every provider
5. **Deployment**: run one script; switch providers in config

## Requirements

- Python 3.10–3.11 (repo-root `.venv`)
- `pip install -r requirements.txt` (aiohttp, websockets, nltk)
- Godot 4.4+
- For the local provider: a prebuilt `llama-server.exe` and a GGUF model —
  llama.cpp is **not** a Python dependency of this project

## Notes

- The unified server approach reduces complexity significantly
- NPC name canonicalization prevents memory fragmentation
- WebSocket streaming provides smooth, real-time responses
- Decisions and dialogue share the configured provider
- The system is designed for easy extension and modification
