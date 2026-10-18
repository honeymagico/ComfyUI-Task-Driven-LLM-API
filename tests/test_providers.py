"""Provider adapters: wire format, error mapping and discovery."""

from __future__ import annotations

import pytest

from conftest import config_module, providers_module

ChatRequest = providers_module.ChatRequest
ImagePart = providers_module.ImagePart
Turn = providers_module.Turn
ProviderError = providers_module.ProviderError


def _request(provider: str, base_url: str, **overrides) -> ChatRequest:
    defaults = dict(
        provider=provider,
        base_url=base_url,
        model="stub-model",
        system="系統提示",
        turns=[Turn(text="第一則", images=[ImagePart(slot="IMAGE1", label="【IMAGE1】", data_b64="QUJD", mime="image/jpeg")])],
        api_key="ollama",
        timeout=10,
    )
    defaults.update(overrides)
    return ChatRequest(**defaults)


# -- URL normalisation ------------------------------------------------------


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("http://127.0.0.1:11434", "http://127.0.0.1:11434/v1/chat/completions"),
        ("http://127.0.0.1:1234/v1", "http://127.0.0.1:1234/v1/chat/completions"),
        ("http://host/v1/", "http://host/v1/chat/completions"),
        ("https://openrouter.ai/api/v1", "https://openrouter.ai/api/v1/chat/completions"),
        ("http://host/v1/chat/completions", "http://host/v1/chat/completions"),
    ],
)
def test_chat_endpoint_normalisation(given, expected):
    assert providers_module.chat_endpoint(given) == expected


def test_models_endpoint_derives_from_the_chat_url():
    assert providers_module.models_endpoint("http://host:1234/v1") == "http://host:1234/v1/models"
    assert providers_module.models_endpoint("http://host:1234/v1/chat/completions") == "http://host:1234/v1/models"


# -- Ollama -----------------------------------------------------------------


def test_ollama_sends_bare_base64_images_and_reports_usage(stub_llm):
    result = providers_module.chat(_request("ollama", stub_llm.url))

    body = stub_llm.state.last("/api/chat")["body"]
    assert body["stream"] is False
    assert body["messages"][0] == {"role": "system", "content": "系統提示"}
    assert body["messages"][1]["images"] == ["QUJD"]
    assert "data:image" not in body["messages"][1]["images"][0]
    assert result.text.startswith("STUB_OLLAMA")
    assert result.usage["eval_count"] == 22
    assert result.duration_ms > 0


def test_ollama_only_sends_think_when_explicitly_set(stub_llm):
    providers_module.chat(_request("ollama", stub_llm.url))
    assert "think" not in stub_llm.state.last("/api/chat")["body"]

    providers_module.chat(_request("ollama", stub_llm.url, think=False))
    assert stub_llm.state.last("/api/chat")["body"]["think"] is False


def test_ollama_retries_once_without_think_on_old_servers(stub_llm):
    stub_llm.state.fail_next_think_unknown_field = True
    result = providers_module.chat(_request("ollama", stub_llm.url, think=True))

    bodies = stub_llm.state.bodies("/api/chat")
    assert len(bodies) == 2
    assert bodies[0]["think"] is True
    assert "think" not in bodies[1]
    assert result.text.startswith("STUB_OLLAMA")


def test_ollama_passes_options_and_keep_alive(stub_llm):
    providers_module.chat(
        _request("ollama", stub_llm.url, temperature=0.25, max_tokens=64, seed=7, keep_alive="10m")
    )
    body = stub_llm.state.last("/api/chat")["body"]
    assert body["options"] == {"temperature": 0.25, "num_predict": 64, "seed": 7}
    assert body["keep_alive"] == "10m"


def test_missing_model_error_tells_the_user_how_to_pull_it(stub_llm):
    stub_llm.state.fail_status = 404
    stub_llm.state.fail_body = {"error": "model 'ghost' not found, try pulling it first"}
    with pytest.raises(ProviderError) as caught:
        providers_module.chat(_request("ollama", stub_llm.url, model="ghost"))
    assert "ghost" in str(caught.value)
    assert "ollama pull" in caught.value.hint


def test_unreachable_server_error_mentions_ollama_serve():
    # Port 9 (discard) is closed on every normal machine.
    with pytest.raises(ProviderError) as caught:
        providers_module.chat(_request("ollama", "http://127.0.0.1:9", timeout=5))
    assert "無法連線" in str(caught.value)
    assert "ollama serve" in caught.value.hint


def test_ollama_list_models(stub_llm):
    names = providers_module.list_models("ollama", {"base_url": stub_llm.url, "timeout": 5})
    assert names == ["stub-text:latest", "stub-vl:latest"]


def test_ollama_probe_reports_version_and_models(stub_llm):
    report = providers_module.probe("ollama", {"base_url": stub_llm.url, "timeout": 5})
    assert report["ok"] is True
    assert report["version"] == "0.0-stub"
    assert len(report["models"]) == 2


# -- OpenAI-compatible ------------------------------------------------------


def test_openai_compatible_builds_content_parts_and_auth_header(stub_llm):
    result = providers_module.chat(
        _request("openai_compatible", stub_llm.openai_url, api_key="secret", temperature=0.3)
    )
    entry = stub_llm.state.last("/v1/chat/completions")
    content = entry["body"]["messages"][1]["content"]
    assert content[0] == {"type": "text", "text": "第一則"}
    assert content[1]["image_url"]["url"] == "data:image/jpeg;base64,QUJD"
    assert entry["headers"]["Authorization"] == "Bearer secret"
    assert entry["body"]["temperature"] == 0.3
    assert result.text == "STUB_OPENAI images=1"
    assert result.usage["total_tokens"] == 11


def test_openai_compatible_system_message_is_plain_text(stub_llm):
    providers_module.chat(_request("openai_compatible", stub_llm.openai_url))
    messages = stub_llm.state.last("/v1/chat/completions")["body"]["messages"]
    assert messages[0] == {"role": "system", "content": "系統提示"}


def test_openai_compatible_omits_auth_header_without_a_key(stub_llm):
    providers_module.chat(_request("openai_compatible", stub_llm.openai_url, api_key=""))
    assert "Authorization" not in stub_llm.state.last("/v1/chat/completions")["headers"]


def test_openai_compatible_401_hint_mentions_the_api_key(stub_llm):
    stub_llm.state.fail_status = 401
    stub_llm.state.fail_body = {"error": {"message": "Invalid API key"}}
    with pytest.raises(ProviderError) as caught:
        providers_module.chat(_request("openai_compatible", stub_llm.openai_url))
    assert "Invalid API key" in str(caught.value)
    assert "API Key" in caught.value.hint


def test_openai_compatible_list_models(stub_llm):
    names = providers_module.list_models("openai_compatible", {"base_url": stub_llm.openai_url, "timeout": 5})
    assert names == ["stub-openai-model"]


# -- registry / helpers -----------------------------------------------------


def test_unknown_provider_is_rejected_with_the_supported_list():
    with pytest.raises(ProviderError) as caught:
        providers_module.get_provider("gemini")
    assert "gemini" in str(caught.value)
    assert "ollama" in caught.value.hint


def test_describe_request_hides_image_bytes(stub_llm):
    described = providers_module.describe_request(_request("ollama", stub_llm.url))
    assert described["messages"][0]["images"][0]["slot"] == "IMAGE1"
    assert "data_b64" not in described["messages"][0]["images"][0]


def test_render_request_preview_shows_tags_and_sizes():
    transcript = providers_module.render_request_preview(
        _request("ollama", "http://x", turns=[Turn(text="【IMAGE1（景）】", images=[ImagePart(slot="IMAGE1", label="", data_b64="QQ==", mime="image/jpeg", role="景", width=64, height=48, byte_count=2048)])])
    )
    assert "═══ system ═══" in transcript
    assert "═══ user #1 ═══" in transcript
    assert "[圖片] IMAGE1（景） — 64×48, 2.0 KB" in transcript
    assert "【IMAGE1（景）】" in transcript


def test_payload_size_hint_counts_images_and_text():
    hint = providers_module.payload_size_hint(_request("ollama", "http://x"))
    assert hint["turns"] == 1
    assert hint["images"] == 1
    assert hint["image_kb"] == 0.0


# -- config -----------------------------------------------------------------


def test_provider_settings_defaults_to_local_ollama(sandbox, monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    settings = config_module.provider_settings("ollama")
    assert settings["base_url"] == "http://127.0.0.1:11434"
    assert settings["api_key"] == "ollama"
    assert settings["timeout"] == 300


def test_env_vars_override_the_config_file(sandbox, monkeypatch):
    monkeypatch.setenv("TASKLLM_OLLAMA_BASE_URL", "http://10.0.0.5:11434")
    monkeypatch.setenv("OLLAMA_HOST", "10.0.0.9:11434")
    settings = config_module.provider_settings("ollama")
    # OLLAMA_HOST wins because it is the convention the ollama CLI follows.
    assert settings["base_url"] == "http://10.0.0.9:11434"

    monkeypatch.delenv("OLLAMA_HOST")
    assert config_module.provider_settings("ollama")["base_url"] == "http://10.0.0.5:11434"


def test_public_config_never_leaks_the_api_key(sandbox):
    config_module.apply_public_update({"providers": {"openai_compatible": {"api_key": "sk-super-secret"}}})
    public = config_module.public_config()
    assert public["providers"]["openai_compatible"]["api_key_set"] is True
    assert "api_key" not in public["providers"]["openai_compatible"]
    assert "sk-super-secret" not in str(public)
    assert public["providers"]["openai_compatible"]["api_key_hint"].endswith("cret")


def test_omitting_api_key_keeps_the_stored_one(sandbox):
    config_module.apply_public_update({"providers": {"openai_compatible": {"api_key": "keep-me"}}})
    config_module.apply_public_update({"providers": {"openai_compatible": {"base_url": "http://new/v1"}}})
    settings = config_module.provider_settings("openai_compatible")
    assert settings["api_key"] == "keep-me"
    assert settings["base_url"] == "http://new/v1"


def test_empty_api_key_clears_it(sandbox):
    config_module.apply_public_update({"providers": {"openai_compatible": {"api_key": "bye"}}})
    config_module.apply_public_update({"providers": {"openai_compatible": {"api_key": ""}}})
    assert config_module.provider_settings("openai_compatible")["api_key"] == ""


def test_base_url_trailing_slash_is_trimmed(sandbox):
    config_module.apply_public_update({"providers": {"ollama": {"base_url": "http://host:11434/"}}})
    assert config_module.provider_settings("ollama")["base_url"] == "http://host:11434"
