"""The editor's HTTP surface.

These drive the aiohttp handlers directly through a fresh
``RouteTableDef`` so the contract the browser depends on -- status codes,
``ok`` flags, masked keys, the preview transcript -- is pinned by tests
rather than by clicking around in ComfyUI.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from conftest import config_module, library_module, models_module, nodes_module, paths

aiohttp_web = pytest.importorskip("aiohttp.web")
import importlib  # noqa: E402

routes_module = importlib.import_module(nodes_module.__name__.rsplit(".", 1)[0] + ".routes")


class FakeRequest:
    """Just enough of aiohttp's Request for the handlers under test."""

    def __init__(self, payload=None):
        self._payload = payload

    async def json(self):
        if self._payload is None:
            raise ValueError("no body")
        return self._payload


def _build_table():
    table = aiohttp_web.RouteTableDef()
    assert routes_module.register_routes(table) is True
    return table


def _handler(table, method: str, path: str):
    for route in table:
        if route.method == method and route.path == path:
            return route.handler
    raise AssertionError(f"找不到路由 {method} {path}")


def _call(table, method: str, path: str, payload=None):
    handler = _handler(table, method, path)
    response = asyncio.run(handler(FakeRequest(payload)))
    body = json.loads(response.body.decode("utf-8")) if response.body else None
    return response.status, body


@pytest.fixture
def table(seeded_library):
    return _build_table()


# -- prompt library ---------------------------------------------------------


def test_get_library_returns_tasks_and_paths(table, seeded_library):
    status, body = _call(table, "GET", "/task_llm/library")
    assert status == 200
    assert body["ok"] is True
    assert len(body["library"]["tasks"]) == len(seeded_library.tasks)
    assert body["tasks"][0]["label"].startswith("提示詞生成 · ")
    assert body["locate"]["tasks_path"].endswith("tasks.json")
    assert body["warnings"] == []


def test_saving_a_library_round_trips_and_reports_labels(table):
    payload = {
        "library": {
            "title": "測試庫",
            "default_system": "你是助手",
            "tasks": [{"id": "a", "title": "新任務", "prompt": "做事", "image_roles": {"IMAGE2": "人"}}],
        }
    }
    status, body = _call(table, "POST", "/task_llm/library", payload)
    assert status == 200
    assert body["ok"] is True
    assert body["labels"] == ["新任務"]

    status, reloaded = _call(table, "GET", "/task_llm/library")
    assert reloaded["library"]["default_system"] == "你是助手"
    assert reloaded["library"]["tasks"][0]["image_roles"] == {"IMAGE2": "人"}


def test_saving_junk_reports_warnings_but_still_saves(table):
    status, body = _call(
        table,
        "POST",
        "/task_llm/library",
        {"library": {"tasks": [{"title": "同名", "prompt": "a"}, {"title": "同名", "prompt": "b"}, "junk"]}},
    )
    assert status == 200
    assert any("重複" in warning for warning in body["warnings"])
    assert body["labels"] == ["同名", "同名 (2)"]


def test_a_library_that_is_not_json_is_rejected_with_a_message(table):
    status, body = _call(table, "POST", "/task_llm/library", {"library": "just a string"})
    assert status == 400
    assert body["ok"] is False
    assert "JSON" in body["error"]


def test_tasks_endpoint_returns_dropdown_labels(table, seeded_library):
    status, body = _call(table, "GET", "/task_llm/tasks")
    assert status == 200
    assert "提示詞生成 · 生活化抓拍照片" in body["labels"]


def test_restore_replaces_the_library_with_the_shipped_examples(table):
    _call(table, "POST", "/task_llm/library", {"library": {"tasks": []}})
    assert _call(table, "GET", "/task_llm/tasks")[1]["labels"] == [
        "（提示詞庫是空的，請按節點上的「提示詞庫」按鈕新增 TASK）"
    ]

    status, body = _call(table, "POST", "/task_llm/library/restore")
    assert status == 200
    assert len(body["library"]["tasks"]) >= 6


def test_import_replaces_and_export_returns_a_download(table):
    status, body = _call(
        table, "POST", "/task_llm/import", {"library": {"tasks": [{"title": "匯入的", "prompt": "p"}]}}
    )
    assert status == 200 and body["labels"] == ["匯入的"]

    export_handler = _handler(table, "GET", "/task_llm/export")
    response = asyncio.run(export_handler(FakeRequest()))
    assert response.headers["Content-Disposition"].endswith('filename="tasks.json"')
    exported = json.loads(response.body.decode("utf-8"))
    assert exported["tasks"][0]["title"] == "匯入的"


# -- provider settings ------------------------------------------------------


def test_get_config_masks_the_api_key(table):
    config_module.apply_public_update({"providers": {"openai_compatible": {"api_key": "sk-hidden"}}})
    status, body = _call(table, "GET", "/task_llm/config")
    assert status == 200
    assert body["config"]["providers"]["openai_compatible"]["api_key_set"] is True
    assert "sk-hidden" not in json.dumps(body)


def test_post_config_keeps_an_untouched_key(table):
    config_module.apply_public_update({"providers": {"ollama": {"api_key": "ollama-secret"}}})
    status, body = _call(
        table, "POST", "/task_llm/config", {"config": {"providers": {"ollama": {"base_url": "http://x:1"}}}}
    )
    assert status == 200
    assert body["config"]["providers"]["ollama"]["base_url"] == "http://x:1"
    assert config_module.provider_settings("ollama")["api_key"] == "ollama-secret"


def test_config_lists_providers_and_presets_for_the_settings_tab(table):
    _status, body = _call(table, "GET", "/task_llm/config")
    config = body["config"]
    assert set(config["providers"]) == {"ollama", "openai_compatible"}
    assert config["provider_info"]["ollama"]["needs_key"] is False
    assert any(preset["label"] == "OpenRouter" for preset in config["base_url_presets"])


# -- discovery --------------------------------------------------------------


def test_models_endpoint_uses_the_stub_server(table, stub_llm, monkeypatch):
    monkeypatch.setattr(
        config_module,
        "provider_settings",
        lambda provider: {"provider": provider, "base_url": stub_llm.url, "api_key": "ollama", "timeout": 5},
    )
    models_module.invalidate()
    status, body = _call(table, "POST", "/task_llm/models", {"provider": "ollama"})
    assert status == 200
    assert body["models"] == ["stub-text:latest", "stub-vl:latest"]


def test_models_endpoint_reports_an_unreachable_server_without_500(table, monkeypatch):
    monkeypatch.setattr(
        config_module,
        "provider_settings",
        lambda provider: {"provider": provider, "base_url": "http://127.0.0.1:9", "api_key": "", "timeout": 5},
    )
    models_module.invalidate()
    status, body = _call(table, "POST", "/task_llm/models", {"provider": "ollama"})
    assert status == 200  # the editor shows the message, it is not an HTTP error
    assert body["ok"] is False
    assert "無法連線" in body["error"]
    assert body["models"] == []


def test_probe_reports_version_and_models(table, stub_llm, monkeypatch):
    monkeypatch.setattr(
        config_module,
        "provider_settings",
        lambda provider: {"provider": provider, "base_url": stub_llm.url, "api_key": "ollama", "timeout": 5},
    )
    status, body = _call(table, "POST", "/task_llm/probe", {"provider": "ollama"})
    assert status == 200
    assert body["ok"] is True
    assert body["version"] == "0.0-stub"


def test_every_endpoint_rejects_an_unknown_provider(table):
    status, body = _call(table, "POST", "/task_llm/probe", {"provider": "gemini"})
    assert status == 400 and "gemini" in body["error"]
    status, body = _call(table, "POST", "/task_llm/models", {"provider": "gemini"})
    assert status == 400 and "gemini" in body["error"]


# -- preview ----------------------------------------------------------------


def test_preview_renders_the_exact_transcript_for_a_wiring(table, seeded_library):
    status, body = _call(
        table,
        "POST",
        "/task_llm/preview",
        {
            "task": "提示詞生成 · 多圖參考合成（景 / 人 / 色調）",
            "connected": ["IMAGE1", "IMAGE3"],
            "user_prompt": "改成夜晚",
        },
    )
    assert status == 200
    assert body["ok"] is True
    transcript = body["transcript"]
    assert "【IMAGE1（景）】" in transcript
    assert "【IMAGE3（色調）】" in transcript
    assert "未連接的接口：IMAGE2" in transcript
    assert "改成夜晚" in transcript
    assert body["used_slots"] == ["IMAGE1", "IMAGE3"]
    assert body["missing_declared"] == ["IMAGE2"]
    assert body["request_size"]["images"] == 2
    assert any("IMAGE2" in note for note in body["notes"])


def test_preview_accepts_an_unsaved_task_body(table, seeded_library):
    """The editor previews what is on screen, not what is on disk."""
    status, body = _call(
        table,
        "POST",
        "/task_llm/preview",
        {
            "task_body": {
                "title": "尚未儲存的 TASK",
                "prompt": "參考 IMAGE2 的人物。",
                "image_roles": {"IMAGE2": "人"},
            },
            "default_system": "這是預覽用的系統提示",
            "connected": ["IMAGE2"],
            "user_prompt": "現場指示",
        },
    )
    assert status == 200
    assert "這是預覽用的系統提示" in body["transcript"]
    assert "【IMAGE2（人）】" in body["transcript"]
    assert "現場指示" in body["transcript"]
    assert body["used_slots"] == ["IMAGE2"]


def test_preview_rejects_an_unusable_task_body(table):
    status, body = _call(table, "POST", "/task_llm/preview", {"task_body": {"title": "", "prompt": ""}})
    assert status == 400
    assert "預覽失敗" in body["error"]


def test_preview_ignores_unknown_slots_and_unknown_tasks(table):
    status, body = _call(
        table, "POST", "/task_llm/preview", {"task": "不存在", "connected": ["IMAGE9"]}
    )
    assert status == 400
    assert "找不到" in body["error"]


def test_preview_of_a_text_only_task_has_no_images(table):
    status, body = _call(
        table, "POST", "/task_llm/preview", {"task": "文字處理 · 提示詞潤飾與英譯", "user_prompt": "一隻貓"}
    )
    assert status == 200
    assert body["used_slots"] == []
    assert "沒有連接任何圖片" in body["transcript"]


# -- filesystem -------------------------------------------------------------


def test_open_folder_returns_the_real_path(table, sandbox):
    status, body = _call(table, "POST", "/task_llm/open_folder", {"target": "prompts"})
    assert status == 200
    assert body["path"] == str(sandbox / "prompts")
    assert body["locate"]["data_root"] == str(sandbox)


def test_open_folder_can_target_the_config_file(table, sandbox):
    status, body = _call(table, "POST", "/task_llm/open_folder", {"target": "config"})
    assert status == 200
    assert body["path"] == str(sandbox / "config.json")


def test_register_routes_reports_false_without_comfyui(monkeypatch):
    # Importing ``server`` must fail for this assertion to mean anything.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "server":
            raise ImportError("no comfyui here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert routes_module.register_routes() is False


def test_library_path_stays_inside_the_sandbox(table, sandbox):
    _call(table, "POST", "/task_llm/library", {"library": {"tasks": [{"title": "x", "prompt": "y"}]}})
    assert paths.tasks_path().parent == sandbox / "prompts"
    assert library_module.load_library()[0].tasks[0].title == "x"
