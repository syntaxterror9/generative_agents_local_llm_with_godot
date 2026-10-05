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
