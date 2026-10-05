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
