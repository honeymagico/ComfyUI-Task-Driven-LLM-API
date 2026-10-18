"""End-to-end node execution against a stub Ollama / OpenAI server."""

from __future__ import annotations

import json

import pytest

from conftest import config_module, library_module, nodes_module

TASK_REFERENCE = "提示詞生成 · 多圖參考合成（景 / 人 / 色調）"
TASK_TEXT_ONLY = "文字處理 · 提示詞潤飾與英譯"


def _settings_factory(stub_llm, **overrides):
    def settings(provider: str):
        base = {
            "provider": provider,
            "base_url": stub_llm.url if provider == "ollama" else stub_llm.openai_url,
            "api_key": "ollama",
            "default_model": "stub-vl:latest",
            "keep_alive": "5m",
            "timeout": 10,
            "image_quality": 90,
        }
        base.update(overrides)
        return base

    return settings


@pytest.fixture
def node_env(seeded_library, stub_llm, monkeypatch):
    monkeypatch.setattr(config_module, "provider_settings", _settings_factory(stub_llm))
    return stub_llm


def _run(node, **kwargs):
    params = {
        "task": TASK_REFERENCE,
        "provider": "ollama",
        "model": "stub-vl:latest",
        "model_custom": "",
        "user_prompt": "",
        "system_override": "",
        "image_delivery": "逐張標註",
        "think": "關閉",
        "temperature": 0.7,
        "num_predict": 1024,
        "seed": 42,
        "image_max_side": 1024,
        "image_format": "JPEG",
        "keep_alive": "5m",
        "timeout": 10,
        "debug": False,
    }
    params.update(kwargs)
    return node.run(**params)


# -- interface --------------------------------------------------------------


def test_input_types_expose_the_library_and_six_optional_sockets(seeded_library):
    inputs = nodes_module.TaskDrivenLLM.INPUT_TYPES()
    assert TASK_REFERENCE in inputs["required"]["task"][0]
    assert list(inputs["optional"]) == ["IMAGE1", "IMAGE2", "IMAGE3", "IMAGE4", "IMAGE5", "IMAGE6"]
    assert inputs["required"]["provider"][0] == ["ollama", "openai_compatible"]
    # A named `seed` widget gets ComfyUI's control_after_generate for free.
    assert "seed" in inputs["required"]


def test_input_types_survive_a_broken_library(sandbox):
    broken = sandbox / "prompts" / "tasks.json"
    broken.parent.mkdir(parents=True, exist_ok=True)
    broken.write_text("not json at all", encoding="utf-8")
    inputs = nodes_module.TaskDrivenLLM.INPUT_TYPES()
    assert inputs["required"]["task"][0]  # still usable, no exception


# -- happy path -------------------------------------------------------------


def test_three_images_are_labelled_by_socket(node_env, image_tensor):
    node = nodes_module.TaskDrivenLLM()
    out = _run(node, IMAGE1=image_tensor(64, 48), IMAGE2=image_tensor(64, 48), IMAGE3=image_tensor(64, 48))

    assert out["result"][0].startswith("STUB_OLLAMA")
    assert out["ui"]["text"][0] == out["result"][0]

    messages = node_env.state.last("/api/chat")["body"]["messages"]
    system = [message for message in messages if message["role"] == "system"]
    user = [message for message in messages if message["role"] == "user"]
    assert system and system[0]["content"].startswith("你是一位專業的視覺分析師")

    texts = [message["content"] for message in user]
    assert texts[0].startswith("【本次輸入圖片對照】")
    assert "- IMAGE1（景）：第 1 張" in texts[0]
    assert "- IMAGE3（色調）：第 3 張" in texts[0]
    assert texts[1] == "【IMAGE1（景）】"
    assert texts[2] == "【IMAGE2（人）】"
    assert texts[3] == "【IMAGE3（色調）】"
    # Exactly one picture per labelled message -- that is what makes the
    # mapping survive any model's chat template.
    assert [len(message.get("images") or []) for message in user] == [0, 1, 1, 1, 0]
    assert "images" not in user[0]
    assert node_env.state.last("/api/chat")["body"]["model"] == "stub-vl:latest"


def test_meta_output_describes_the_run(node_env, image_tensor):
    node = nodes_module.TaskDrivenLLM()
    out = _run(node, IMAGE1=image_tensor(64, 48), user_prompt="黃昏的市場")
    meta = json.loads(out["result"][2])

    assert meta["task"]["id"] == "multi-reference-compose"
    assert meta["provider"] == "ollama"
    assert meta["model"] == "stub-vl:latest"
    assert [image["slot"] for image in meta["images"]] == ["IMAGE1"]
    assert meta["images"][0]["role"] == "景"
    assert meta["request_size"]["images"] == 1
    assert meta["missing_declared_images"] == ["IMAGE2", "IMAGE3"]
    assert meta["usage"]["eval_count"] == 22
    assert any(
        "使用者補充指示" in message.get("content", "")
        for message in node_env.state.last("/api/chat")["body"]["messages"]
    )


def test_task_level_overrides_win_over_the_node_widgets(node_env, image_tensor):
    node = nodes_module.TaskDrivenLLM()
    _run(node, temperature=0.1, num_predict=9999, think="開啟", IMAGE1=image_tensor(8, 8))

    body = node_env.state.last("/api/chat")["body"]
    # The reference task ships temperature 0.85 / num_predict 900 / think false.
    assert body["options"]["temperature"] == 0.85
    assert body["options"]["num_predict"] == 900
    assert body["think"] is False
    assert body["options"]["seed"] == 42


def test_zero_image_task_runs_without_any_image(node_env):
    node = nodes_module.TaskDrivenLLM()
    out = _run(node, task=TASK_TEXT_ONLY, user_prompt="一隻戴帽子的貓")
    body = node_env.state.last("/api/chat")["body"]
    assert all("images" not in message for message in body["messages"])
    assert "一隻戴帽子的貓" in body["messages"][-1]["content"]
    assert out["result"][0].startswith("STUB_OLLAMA")


def test_model_custom_overrides_the_dropdown(node_env, image_tensor):
    node = nodes_module.TaskDrivenLLM()
    _run(node, model="（未偵測到模型）", model_custom="my-own:tag", IMAGE1=image_tensor(8, 8))
    assert node_env.state.last("/api/chat")["body"]["model"] == "my-own:tag"


def test_images_are_downscaled_before_sending(node_env, image_tensor):
    node = nodes_module.TaskDrivenLLM()
    out = _run(node, IMAGE1=image_tensor(1024, 512), image_max_side=128)
    meta = json.loads(out["result"][2])
    assert (meta["images"][0]["width"], meta["images"][0]["height"]) == (64, 128)
    assert meta["images"][0]["source_width"] == 512
    assert meta["images"][0]["downscaled"] is True


def test_debug_flag_embeds_the_transcript_in_meta(node_env, image_tensor):
    node = nodes_module.TaskDrivenLLM()
    out = _run(node, debug=True, IMAGE1=image_tensor(8, 8))
    meta = json.loads(out["result"][2])
    assert "request_preview" in meta
    assert "【IMAGE1（景）】" in meta["request_preview"]


def test_single_message_delivery_puts_everything_in_one_message(node_env, image_tensor):
    node = nodes_module.TaskDrivenLLM()
    _run(
        node,
        image_delivery="集中單一訊息",
        IMAGE1=image_tensor(8, 8),
        IMAGE2=image_tensor(8, 8),
    )
    messages = node_env.state.last("/api/chat")["body"]["messages"]
    user_messages = [message for message in messages if message["role"] == "user"]
    assert len(user_messages) == 1
    assert len(user_messages[0]["images"]) == 2


def test_openai_compatible_provider_end_to_end(node_env, image_tensor):
    node = nodes_module.TaskDrivenLLM()
    out = _run(node, provider="openai_compatible", IMAGE1=image_tensor(8, 8))
    messages = node_env.state.last("/v1/chat/completions")["body"]["messages"]
    user_messages = [message for message in messages if message["role"] == "user"]
    # [0] is the manifest (text only), [1] is the labelled IMAGE1 message.
    assert user_messages[0]["content"][0]["type"] == "text"
    assert user_messages[1]["content"][0]["text"] == "【IMAGE1（景）】"
    assert user_messages[1]["content"][1]["type"] == "image_url"
    assert out["result"][0] == "STUB_OPENAI images=1"


def test_system_override_reaches_the_wire(node_env, image_tensor):
    node = nodes_module.TaskDrivenLLM()
    _run(node, system_override="只有這句話", IMAGE1=image_tensor(8, 8))
    assert node_env.state.last("/api/chat")["body"]["messages"][0]["content"] == "只有這句話"


# -- failure paths ----------------------------------------------------------


def test_unknown_task_raises_an_actionable_error(node_env):
    node = nodes_module.TaskDrivenLLM()
    with pytest.raises(RuntimeError) as caught:
        _run(node, task="不存在的 TASK")
    message = str(caught.value)
    assert "找不到 TASK" in message
    assert "提示詞庫" in message


def test_no_model_selected_raises(node_env, monkeypatch, image_tensor):
    monkeypatch.setattr(
        config_module,
        "provider_settings",
        lambda provider: {
            "provider": provider,
            "base_url": node_env.url,
            "api_key": "ollama",
            "default_model": "",
            "timeout": 10,
            "image_quality": 90,
        },
    )
    node = nodes_module.TaskDrivenLLM()
    with pytest.raises(RuntimeError) as caught:
        _run(node, model="", model_custom="", IMAGE1=image_tensor(8, 8))
    assert "沒有選擇模型" in str(caught.value)


def test_provider_error_becomes_a_readable_runtime_error(node_env, image_tensor):
    node_env.state.fail_status = 404
    node_env.state.fail_body = {"error": "model 'stub-vl:latest' not found"}
    node = nodes_module.TaskDrivenLLM()
    with pytest.raises(RuntimeError) as caught:
        _run(node, IMAGE1=image_tensor(8, 8))
    assert "ollama pull" in str(caught.value)


def test_unknown_provider_is_rejected(node_env, image_tensor):
    node = nodes_module.TaskDrivenLLM()
    with pytest.raises(RuntimeError) as caught:
        _run(node, provider="gemini", IMAGE1=image_tensor(8, 8))


def test_a_task_pinned_to_another_provider_does_not_hijack_the_model(node_env, sandbox, monkeypatch, image_tensor):
    library, _ = library_module.coerce_library(
        {
            "tasks": [
                {
                    "title": "雲端任務",
                    "prompt": "p",
                    "provider": "openai_compatible",
                    "model": "gpt-4o-mini",
                }
            ]
        }
    )
    library_module.save_library(library)
    monkeypatch.setattr(config_module, "provider_settings", _settings_factory(node_env))

    node = nodes_module.TaskDrivenLLM()
    _run(node, task="雲端任務", provider="ollama", model="stub-vl:latest", IMAGE1=image_tensor(8, 8))

    # The task pins the provider, so the call must land on the OpenAI route
    # and must NOT reuse the Ollama model the node widget still shows.
    body = node_env.state.last("/v1/chat/completions")["body"]
    assert body["model"] == "gpt-4o-mini"
