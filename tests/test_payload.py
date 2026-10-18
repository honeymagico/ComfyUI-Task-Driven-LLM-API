"""The IMAGE1…IMAGE6 disambiguation contract.

These are the tests that matter most: if the payload builder ever lets the
model confuse which picture came from which socket, the whole node is
useless for multi-reference work.
"""

from __future__ import annotations

from conftest import library_module, payload_module, providers_module


def _image(slot: str, role: str = "") -> providers_module.ImagePart:
    return providers_module.ImagePart(
        slot=slot, label="", data_b64="AAAA", mime="image/jpeg", role=role, width=64, height=48, byte_count=100
    )


def _task(**overrides):
    data = {"title": "T", "prompt": "請依照指示完成任務。"}
    data.update(overrides)
    library, _ = library_module.coerce_library({"tasks": [data]})
    return library.tasks[0]


def test_per_image_delivery_labels_every_image(seeded_library):
    task = seeded_library.resolve("提示詞生成 · 多圖參考合成（景 / 人 / 色調）")
    built = payload_module.build_payload(
        task,
        seeded_library,
        {"IMAGE1": _image("IMAGE1", "景"), "IMAGE2": _image("IMAGE2", "人"), "IMAGE3": _image("IMAGE3", "色調")},
    )

    texts = [turn.text for turn in built.turns]
    assert texts[0].startswith("【本次輸入圖片對照】")
    assert "- IMAGE1（景）：第 1 張" in texts[0]
    assert "- IMAGE2（人）：第 2 張" in texts[0]
    assert "- IMAGE3（色調）：第 3 張" in texts[0]
    assert texts[1] == "【IMAGE1（景）】"
    assert texts[2] == "【IMAGE2（人）】"
    assert texts[3] == "【IMAGE3（色調）】"
    assert texts[4] == task.prompt

    # Each labelled turn owns exactly one image, so no chat template can
    # reorder text and pictures into an ambiguous pair.
    assert [len(turn.images) for turn in built.turns] == [0, 1, 1, 1, 0]
    assert [image.slot for image in built.turns[2].images] == ["IMAGE2"]


def test_manifest_names_the_sockets_the_task_expected_but_got_none(seeded_library):
    task = seeded_library.resolve("提示詞生成 · 多圖參考合成（景 / 人 / 色調）")
    built = payload_module.build_payload(task, seeded_library, {"IMAGE1": _image("IMAGE1", "景")})

    manifest = built.turns[0].text
    assert "- IMAGE1（景）：第 1 張" in manifest
    assert "未連接的接口：IMAGE2、IMAGE3" in manifest
    assert "不要臆測" in manifest
    assert built.missing_declared == ["IMAGE2", "IMAGE3"]
    assert any("IMAGE2" in note for note in built.notes)


def test_zero_images_still_produces_a_valid_request(seeded_library):
    task = seeded_library.resolve("文字處理 · 提示詞潤飾與英譯")
    built = payload_module.build_payload(task, seeded_library, {}, user_prompt="一隻貓")

    assert built.turns[0].text.startswith("【本次輸入圖片對照】")
    assert "沒有連接任何圖片" in built.turns[0].text
    assert all(not turn.images for turn in built.turns)
    assert built.used_slots == []
    # user_prompt is appended as its own turn when the task has no placeholder.
    assert built.turns[-1].text == "【使用者補充指示】\n一隻貓"


def test_user_prompt_token_is_substituted_in_place(sandbox):
    task = _task(prompt="請照這個做：\n{{user_prompt}}\n以上。")
    library, _ = library_module.coerce_library({"tasks": [{"title": "P", "prompt": "x"}]})
    built = payload_module.build_payload(task, library, {}, user_prompt="畫一隻狗")

    joined = "\n".join(turn.text for turn in built.turns)
    assert "畫一隻狗" in joined
    assert "{{user_prompt}}" not in joined
    # Substituted, therefore not appended a second time.
    assert "使用者補充指示" not in joined


def test_empty_user_prompt_with_token_is_reported(sandbox):
    task = _task(prompt="請照這個做：{{user_prompt}}")
    library, _ = library_module.coerce_library({"tasks": [{"title": "P", "prompt": "x"}]})
    built = payload_module.build_payload(task, library, {}, user_prompt="")
    assert any("user_prompt" in note for note in built.notes)


def test_append_user_prompt_can_be_disabled(sandbox):
    task = _task(append_user_prompt=False)
    library, _ = library_module.coerce_library({"tasks": [{"title": "P", "prompt": "x"}]})
    built = payload_module.build_payload(task, library, {}, user_prompt="不要出現")
    assert "不要出現" not in "\n".join(turn.text for turn in built.turns)


def test_single_message_delivery_keeps_order_in_one_turn(seeded_library):
    task = seeded_library.resolve("提示詞生成 · 多圖元素融合（主體 / 服裝 / 場景 / 風格）")
    images = {f"IMAGE{i}": _image(f"IMAGE{i}") for i in range(1, 5)}
    built = payload_module.build_payload(
        task, seeded_library, images, delivery=payload_module.DELIVERY_SINGLE
    )

    assert len(built.turns) == 1
    assert [image.slot for image in built.turns[0].images] == ["IMAGE1", "IMAGE2", "IMAGE3", "IMAGE4"]
    assert built.turns[0].text.startswith("【本次輸入圖片對照】")
    assert task.prompt in built.turns[0].text


def test_system_prompt_override_wins(seeded_library):
    task = seeded_library.resolve("文字處理 · 提示詞潤飾與英譯")
    built = payload_module.build_payload(task, seeded_library, {}, system_override="新的系統提示")
    assert built.system == "新的系統提示"

    inherited = payload_module.build_payload(task, seeded_library, {})
    assert inherited.system == seeded_library.default_system


def test_images_are_sent_in_socket_order_not_dict_order(seeded_library):
    task = seeded_library.resolve("提示詞生成 · 多圖參考合成（景 / 人 / 色調）")
    images = {
        "IMAGE3": _image("IMAGE3", "色調"),
        "IMAGE1": _image("IMAGE1", "景"),
    }
    built = payload_module.build_payload(task, seeded_library, images)
    assert built.used_slots == ["IMAGE1", "IMAGE3"]
    assert "- IMAGE3（色調）：第 2 張" in built.turns[0].text


def test_task_without_prompt_falls_back_to_a_generic_instruction(sandbox):
    task = _task(prompt="", system="你是助手")
    library, _ = library_module.coerce_library({"tasks": [{"title": "P", "prompt": "x"}]})
    built = payload_module.build_payload(task, library, {})
    assert built.turns[-1].text == "請依照上述圖片完成任務。"
