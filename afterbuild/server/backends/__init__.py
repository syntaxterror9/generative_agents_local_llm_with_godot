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
