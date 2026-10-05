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
