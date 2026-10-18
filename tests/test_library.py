"""The prompt library must be lenient on read and lossless on write."""

from __future__ import annotations

import json

from conftest import library_module, paths


def test_seeded_library_loads_without_warnings(seeded_library):
    labels = [task.label for task in seeded_library.tasks]
    assert "提示詞生成 · 多圖參考合成（景 / 人 / 色調）" in labels
    assert "提示詞生成 · 生活化抓拍照片" in labels
    first = seeded_library.resolve("提示詞生成 · 多圖參考合成（景 / 人 / 色調）")
    assert first.role_for("IMAGE1") == "景"
    assert first.declared_slots() == ["IMAGE1", "IMAGE2", "IMAGE3"]


def test_bare_list_is_accepted(sandbox):
    library, warnings = library_module.coerce_library(
        [{"title": "只有標題", "prompt": "做點事"}]
    )
    assert warnings == []
    assert library.tasks[0].label == "只有標題"
    assert library.default_system == library_module.DEFAULT_SYSTEM


def test_duplicate_titles_are_disambiguated(sandbox):
    library, warnings = library_module.coerce_library(
        {"tasks": [{"title": "同名", "prompt": "a"}, {"title": "同名", "prompt": "b"}]}
    )
    assert [task.label for task in library.tasks] == ["同名", "同名 (2)"]
    assert any("重複" in warning for warning in warnings)
    # Ids must stay unique too, or workflow lookups would be ambiguous.
    assert library.tasks[0].id != library.tasks[1].id


def test_non_ascii_titles_get_stable_ids(sandbox):
    library, _ = library_module.coerce_library({"tasks": [{"title": "看圖說故事", "prompt": "x"}]})
    assert library.tasks[0].id == "看圖說故事"


def test_unknown_image_roles_are_dropped_with_a_warning(sandbox):
    library, warnings = library_module.coerce_library(
        {"tasks": [{"title": "T", "prompt": "p", "image_roles": {"IMAGE1": "景", "IMAGE9": "不存在"}}]}
    )
    assert library.tasks[0].image_roles == {"IMAGE1": "景"}
    assert any("IMAGE9" in warning for warning in warnings)


def test_junk_entries_are_skipped_not_fatal(sandbox):
    library, warnings = library_module.coerce_library(
        {"tasks": ["not an object", {}, {"title": "好的", "prompt": "p"}]}
    )
    assert [task.label for task in library.tasks] == ["好的"]
    assert len(warnings) >= 1


def test_save_and_reload_round_trip(sandbox):
    library, _ = library_module.coerce_library(
        {
            "title": "我的庫",
            "default_system": "你是助手",
            "tasks": [
                {
                    "id": "t1",
                    "title": "任務一",
                    "category": "測試",
                    "prompt": "做 {something}",
                    "temperature": 1.5,
                    "num_predict": 256,
                    "think": True,
                    "image_roles": {"IMAGE2": "人"},
                }
            ],
        }
    )
    path = library_module.save_library(library)
    assert path.exists()

    reloaded, warnings = library_module.load_library()
    assert warnings == []
    assert reloaded.title == "我的庫"
    assert reloaded.default_system == "你是助手"
    task = reloaded.tasks[0]
    assert (task.label, task.temperature, task.num_predict, task.think) == ("測試 · 任務一", 1.5, 256, True)
    assert task.image_roles == {"IMAGE2": "人"}


def test_save_keeps_a_backup_of_the_previous_version(sandbox):
    library, _ = library_module.coerce_library({"tasks": [{"title": "A", "prompt": "1"}]})
    library_module.save_library(library)
    library.tasks[0].prompt = "2"
    path = library_module.save_library(library)

    backup = path.with_suffix(path.suffix + ".bak")
    assert backup.exists()
    assert json.loads(backup.read_text(encoding="utf-8"))["tasks"][0]["prompt"] == "1"
    assert json.loads(path.read_text(encoding="utf-8"))["tasks"][0]["prompt"] == "2"


def test_missing_library_is_created_from_the_shipped_default(sandbox):
    source = paths.NODE_DIR / "prompts" / "tasks.default.json"
    target = sandbox / "prompts"
    target.mkdir(parents=True, exist_ok=True)
    (target / "tasks.default.json").write_bytes(source.read_bytes())

    assert not (target / "tasks.json").exists()
    library, _ = library_module.load_library()
    assert library.tasks, "應該從 tasks.default.json 播種"
    assert (target / "tasks.json").exists()


def test_empty_library_yields_an_actionable_dropdown(sandbox):
    paths.atomic_write_json(sandbox / "prompts" / "tasks.json", {"tasks": []})
    labels = library_module.task_labels()
    assert len(labels) == 1
    assert "提示詞庫" in labels[0]


def test_corrupt_library_never_raises(sandbox):
    broken = sandbox / "prompts" / "tasks.json"
    broken.parent.mkdir(parents=True, exist_ok=True)
    broken.write_text("{ this is not json", encoding="utf-8")
    labels = library_module.task_labels()
    assert len(labels) == 1
    assert labels[0].startswith("（")
    assert "讀取提示詞庫失敗" in labels[0]


def test_system_prompt_inheritance(sandbox):
    library, _ = library_module.coerce_library(
        {
            "default_system": "全域",
            "tasks": [{"title": "繼承", "prompt": "p"}, {"title": "覆寫", "prompt": "p", "system": "自己的"}],
        }
    )
    assert library.system_for(library.tasks[0]) == "全域"
    assert library.system_for(library.tasks[1]) == "自己的"
