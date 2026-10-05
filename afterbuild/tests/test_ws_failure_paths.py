import asyncio
import json
import socket
from pathlib import Path

import pytest
import websockets

from backends.base import Backend, BackendError
from dialogue_server import DialogueServer


class FailingBackend(Backend):
    """Streams two tokens, then fails like a mid-stream provider error."""

    name = "failing"
    model = "boom"

    async def stream_chat(self, messages, params):
        yield "Par"
        yield "tial"
        raise BackendError("provider exploded")


def free_port() -> int:
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


def write_configs(tmp_path: Path, port: int):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "provider": "echo",
        "websocket_port": port,
        "max_tokens": 50,
        "temperature": 0.7,
        "top_p": 0.9,
    }), encoding="utf-8")
    decision = tmp_path / "decision_config.json"
    decision.write_text(json.dumps({
        "valid_actions": ["idle"],
        "npc_specific_actions": {"default": ["idle"]},
        "npc_specific_targets": {"default": ["self"]},
        "npc_prompts": {"default": {"template": "{npc} {actions} {targets} {context}"}},
        "system_prompts": {"strict": "decision bot"},
        "generation_params": {"max_tokens": 15, "temperature": 0.3, "top_p": 0.5},
    }), encoding="utf-8")
    return config, decision


@pytest.fixture
async def failing_server(tmp_path):
    port = free_port()
    config, decision = write_configs(tmp_path, port)
    server = DialogueServer(
        config_path=str(config),
        decision_config_path=str(decision),
        memory_dir=str(tmp_path / "memories"),
        decision_log_dir=str(tmp_path / "logs"),
    )
    server.backend = FailingBackend()
    task = asyncio.create_task(server.start_server())
    await asyncio.sleep(0.3)
    try:
        yield port
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


async def test_dialogue_failure_sends_error_then_complete_with_partial(failing_server):
    async with websockets.connect(f"ws://127.0.0.1:{failing_server}") as ws:
        await ws.send(json.dumps({"npc": "Leonardo", "message": "Leonardo|Hi"}))
        frames = []
        while True:
            frame = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
            frames.append(frame)
            if frame["type"] == "complete":
                break
        kinds = [f["type"] for f in frames]
        assert kinds.count("token") == 2
        assert "error" in kinds
        assert kinds[-1] == "complete"
        assert frames[-1]["content"] == "Partial"  # partial text, not the fallback
        error_frame = next(f for f in frames if f["type"] == "error")
        assert "provider exploded" in error_frame["content"]


async def test_decision_failure_reports_raw_error(failing_server):
    async with websockets.connect(f"ws://127.0.0.1:{failing_server}") as ws:
        await ws.send(json.dumps({"type": "decision", "npc": "Leonardo", "context": "test"}))
        frame = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
        assert frame["type"] == "decision_result"
        assert frame["action"] == "idle"
        assert frame["target"] == "self"
        assert frame["valid"] is False
        assert frame["debug"]["raw"] == "error"
