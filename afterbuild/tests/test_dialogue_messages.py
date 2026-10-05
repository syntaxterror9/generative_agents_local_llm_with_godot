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


def test_npc_to_npc_memories_are_not_replayed_as_user_turns(tmp_path):
    server = make_server(tmp_path)
    server.memory_cache["Leonardo"] = [
        memory("user said this", "leonardo replied"),
        {"user_input": "einstein said this", "npc_response": "leonardo answered",
         "keywords": ["chat"], "speaker": "Einstein", "importance": 3.0,
         "interaction_type": "npc_to_npc"},
        {"user_input": "legacy entry", "npc_response": "legacy reply",
         "keywords": ["chat"], "speaker": "user", "importance": 3.0},
    ]
    messages = server.build_dialogue_messages("Leonardo", "user", "next")
    turn_contents = [m.content for m in messages[1:]]
    assert "einstein said this" not in turn_contents
    assert "leonardo answered" not in turn_contents
    assert "user said this" in turn_contents
    assert "legacy entry" in turn_contents  # entries without the field still replay


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
