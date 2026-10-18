"""The Task-Driven LLM API node.

One node, one job: take a TASK from the prompt library, attach the images
the user wired into ``IMAGE1`` … ``IMAGE6``, call an LLM API, and return the
answer as text.  The node owns no prompt content of its own -- everything
editable lives in the library, which the user maintains from the in-canvas
editor.

Execution flow::

    task label ──▶ Library.resolve ──▶ Task
    IMAGE1..6  ──▶ imaging.encode_image ──▶ ImagePart(slot, tag)
    Task + images ──▶ payload.build_payload ──▶ system + turns
    ChatRequest ──▶ providers.chat ──▶ text / thinking

Failures are raised as ``RuntimeError`` with an actionable Chinese message
(a missing model says how to pull it, an unreachable server says how to
start it) because ComfyUI surfaces the exception text in the UI verbatim.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from .taskllm import config as config_module
from .taskllm import imaging
from .taskllm import library as library_module
from .taskllm import models as models_module
from .taskllm import payload as payload_module
from .taskllm import providers as providers_module
from .taskllm import textutils
from .taskllm.library import IMAGE_SLOTS

logger = logging.getLogger("task_driven_llm")

NODE_CLASS = "TaskDrivenLLM"
CATEGORY = "Task-Driven LLM"

#: `think` is a tri-state on purpose: older Ollama builds reject the field,
#: so "自動" means "do not send it at all".
THINK_CHOICES = ("關閉", "自動", "開啟")
_THINK_MAP: Dict[str, Optional[bool]] = {"關閉": False, "自動": None, "開啟": True}

_DEBUG_PREFIX = "[Task-Driven LLM]"


def _think_value(choice: Any) -> Optional[bool]:
    if isinstance(choice, bool):
        return choice
    return _THINK_MAP.get(str(choice or "").strip(), False)


def _build_input_types() -> Dict[str, Any]:
    labels = library_module.task_labels()
    default_provider = config_module.load_config().get("default_provider", config_module.PROVIDER_OLLAMA)
    if default_provider not in config_module.PROVIDER_IDS:
        default_provider = config_module.PROVIDER_OLLAMA
    models = models_module.combo_choices(default_provider)

    required: Dict[str, Any] = {
        "task": (
            labels,
            {
                "default": labels[0],
                "tooltip": "要執行的 TASK。清單來自提示詞庫（節點上的「提示詞庫」按鈕可新增／編輯）。",
            },
        ),
        "provider": (
            list(config_module.PROVIDER_IDS),
            {
                "default": default_provider,
                "tooltip": "呼叫哪一種 API。位址與金鑰在「提示詞庫 → 連線設定」裡設定。",
            },
        ),
        "model": (
            models,
            {
                "default": models[0],
                "tooltip": "模型清單由伺服器即時偵測；要填清單以外的名稱請用 model_custom。",
            },
        ),
        "model_custom": (
            "STRING",
            {
                "default": "",
                "tooltip": "非空時優先於上面的模型下拉，可直接輸入任何模型名稱。",
            },
        ),
        "user_prompt": (
            "STRING",
            {
                "default": "",
                "multiline": True,
                "dynamicPrompts": False,
                "tooltip": "這次要交給 LLM 的臨時指示，會附加在 TASK 內容之後（TASK 若已使用 {{user_prompt}} 則改為填入該處）。",
            },
        ),
        "system_override": (
            "STRING",
            {
                "default": "",
                "multiline": True,
                "dynamicPrompts": False,
                "tooltip": "留空＝使用 TASK（或提示詞庫）的 system 提示詞；填寫則整段取代，方便臨時實驗。",
            },
        ),
        "image_delivery": (
            list(payload_module.DELIVERY_MODES),
            {
                "default": payload_module.DELIVERY_PER_IMAGE,
                "tooltip": "「逐張標註」每張圖片單獨一則訊息並附上【IMAGEn】標記，最能避免模型認錯圖；「集中單一訊息」把所有圖片放在同一則訊息，較省 token 但少數模型會混淆順序。",
            },
        ),
        "think": (
            list(THINK_CHOICES),
            {
                "default": "關閉",
                "tooltip": "推理模型的思考模式（僅 Ollama 支援）。生圖提示詞通常不需要思考，關閉可省時間；舊版 Ollama 不認得這個參數，請選「自動」。",
            },
        ),
        "temperature": (
            "FLOAT",
            {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "越高越有創意，越低越穩定。TASK 可覆寫。"},
        ),
        "num_predict": (
            "INT",
            {
                "default": 1024,
                "min": 0,
                "max": 32768,
                "step": 64,
                "tooltip": "最多產生多少 token，0＝不限制。TASK 可覆寫。",
            },
        ),
        "seed": (
            "INT",
            {
                "default": 0,
                "min": 0,
                "max": 0xFFFFFFFFFFFFFFFF,
                "tooltip": "取樣種子；搭配右側的 control_after_generate 使用。",
            },
        ),
        "image_max_side": (
            "INT",
            {
                "default": 1024,
                "min": 0,
                "max": 8192,
                "step": 64,
                "tooltip": "送出前把圖片的長邊縮到這個大小（0＝保持原尺寸）。縮圖能大幅降低 token 與等待時間。",
            },
        ),
        "image_format": (
            ["JPEG", "PNG"],
            {"default": "JPEG", "tooltip": "編碼格式。JPEG 較小；需要精確顏色或線稿時用 PNG。"},
        ),
        "keep_alive": (
            "STRING",
            {"default": "5m", "tooltip": "Ollama 專用：模型在記憶體中保留多久（例如 5m、30s、-1 表示常駐）。"},
        ),
        "timeout": (
            "INT",
            {"default": 300, "min": 5, "max": 3600, "step": 5, "tooltip": "等待伺服器回應的秒數上限。"},
        ),
        "debug": (
            "BOOLEAN",
            {"default": False, "tooltip": "把實際送出的完整內容（含圖片對照表）印到 ComfyUI 主控台，並寫進 meta 輸出。"},
        ),
    }

    optional = {
        slot: ("IMAGE", {"tooltip": f"{slot}：可選輸入。TASK 內容直接寫 {slot} 來引用這張圖。"})
        for slot in IMAGE_SLOTS
    }
    return {"required": required, "optional": optional}


class TaskDrivenLLM:
    """Run a stored TASK against an LLM API with 0..6 images attached."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        try:
            return _build_input_types()
        except Exception as error:  # pragma: no cover - must never break startup
            logger.exception("%s 建立輸入介面失敗：%s", _DEBUG_PREFIX, error)
            fallback = ["（提示詞庫載入失敗，請檢查 prompts/tasks.json）"]
            return {
                "required": {
                    "task": (fallback, {"default": fallback[0]}),
                    "provider": (list(config_module.PROVIDER_IDS), {"default": config_module.PROVIDER_OLLAMA}),
                    "model": (["（模型清單載入失敗）"], {"default": "（模型清單載入失敗）"}),
                    "model_custom": ("STRING", {"default": ""}),
                    "user_prompt": ("STRING", {"default": "", "multiline": True}),
                    "system_override": ("STRING", {"default": "", "multiline": True}),
                    "image_delivery": (list(payload_module.DELIVERY_MODES), {}),
                    "think": (list(THINK_CHOICES), {"default": "關閉"}),
                    "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                    "num_predict": ("INT", {"default": 1024, "min": 0, "max": 32768, "step": 64}),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                    "image_max_side": ("INT", {"default": 1024, "min": 0, "max": 8192, "step": 64}),
                    "image_format": (["JPEG", "PNG"], {"default": "JPEG"}),
                    "keep_alive": ("STRING", {"default": "5m"}),
                    "timeout": ("INT", {"default": 300, "min": 5, "max": 3600, "step": 5}),
                    "debug": ("BOOLEAN", {"default": False}),
                },
                "optional": {slot: ("IMAGE", {}) for slot in IMAGE_SLOTS},
            }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("text", "thinking", "meta")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    DESCRIPTION = (
        "從提示詞庫挑一個 TASK，附加 0～6 張圖片（介面名稱 IMAGE1…IMAGE6），呼叫 Ollama 或 OpenAI 相容 API，"
        "回傳 LLM 產生的文字。圖片會連同【IMAGEn】標記一起送出，模型因此能分辨哪張圖來自哪個接口。"
    )

    # -- main -----------------------------------------------------------------

    def run(
        self,
        task: str,
        provider: str = config_module.PROVIDER_OLLAMA,
        model: str = "",
        model_custom: str = "",
        user_prompt: str = "",
        system_override: str = "",
        image_delivery: str = payload_module.DELIVERY_PER_IMAGE,
        think: str = "關閉",
        temperature: float = 0.7,
        num_predict: int = 1024,
        seed: int = 0,
        image_max_side: int = 1024,
        image_format: str = "JPEG",
        keep_alive: str = "5m",
        timeout: int = 300,
        debug: bool = False,
        **image_inputs: Any,
    ) -> Dict[str, Any]:
        library, _warnings = library_module.load_library()
        selected = library.resolve(task)
        if selected is None:
            available = "、".join(library.labels()[:8]) or "（目前沒有任何 TASK）"
            raise RuntimeError(
                f"提示詞庫裡找不到 TASK「{task}」。\n"
                f"→ 可能是提示詞庫被修改過。目前可用的 TASK：{available}\n"
                "→ 請在節點的「提示詞庫」面板重新選擇一個 TASK。"
            )

        if provider not in config_module.PROVIDER_IDS:
            raise RuntimeError(
                f"不認識的 provider「{provider}」。\n→ 支援：{'、'.join(config_module.PROVIDER_IDS)}"
            )

        effective_provider = (selected.provider or provider) if selected.provider in config_module.PROVIDER_IDS else provider
        settings = config_module.provider_settings(effective_provider)

        if effective_provider != provider:
            logger.info(
                "%s TASK「%s」指定了 provider=%s，覆寫節點上的 %s。",
                _DEBUG_PREFIX, selected.title, effective_provider, provider,
            )

        resolved_model = self._resolve_model(selected, effective_provider, settings, model, model_custom)

        images = self._encode_images(
            image_inputs,
            task=selected,
            fmt=image_format,
            max_side=image_max_side,
            quality=config_module.image_defaults()["quality"],
        )

        built = payload_module.build_payload(
            selected,
            library,
            images,
            user_prompt=user_prompt,
            system_override=system_override,
            delivery=image_delivery,
        )

        request = built.to_chat_request(
            provider=effective_provider,
            base_url=settings.get("base_url", ""),
            model=resolved_model,
            api_key=settings.get("api_key", ""),
            temperature=selected.temperature if selected.temperature is not None else float(temperature),
            max_tokens=selected.num_predict if selected.num_predict is not None else int(num_predict),
            seed=int(seed),
            think=selected.think if selected.think is not None else _think_value(think),
            keep_alive=selected.keep_alive or (keep_alive or None),
            timeout=int(timeout),
        )

        if debug:
            preview = providers_module.render_request_preview(request)
            logger.info("%s 實際送出的內容：\n%s", _DEBUG_PREFIX, preview)
            print(f"{_DEBUG_PREFIX} 送出的完整內容\n{preview}", flush=True)

        try:
            result = providers_module.chat(request)
        except providers_module.ProviderError as error:
            raise RuntimeError(str(error)) from None

        text = textutils.clean_output(result.text)
        thinking = textutils.strip_reasoning(result.thinking)
        if not text and thinking:
            # A reasoning model that never left its thinking block: better to
            # hand back the reasoning than an empty string.
            text = thinking
            built.notes.append("模型只回傳了思考內容，已直接作為輸出（可把 think 設為「關閉」）。")
        if not text:
            built.notes.append("模型回傳了空字串。可提高 num_predict，或檢查模型的視覺能力。")

        meta = self._build_meta(
            request,
            result,
            selected,
            built,
            images,
            extra_notes=built.notes,
            curated_chars=len(text),
            debug=bool(debug),
        )

        logger.info(
            "%s TASK=%s provider=%s model=%s images=%d → %d 字（%.1fs）",
            _DEBUG_PREFIX,
            selected.title,
            effective_provider,
            result.model or resolved_model,
            len(images),
            len(text),
            result.duration_ms / 1000.0,
        )

        return {
            "ui": {"text": [text], "thinking": [thinking], "meta": [json.dumps(meta, ensure_ascii=False, indent=2)]},
            "result": (text, thinking, json.dumps(meta, ensure_ascii=False, indent=2)),
        }

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _resolve_model(
        selected: Any,
        provider: str,
        settings: Dict[str, Any],
        model: str,
        model_custom: str,
    ) -> str:
        # A task pinned to another provider must not dictate this provider's model.
        task_model = selected.model if (selected.model and selected.provider in (None, provider)) else ""
        resolved = task_model or models_module.resolve_model(provider, settings, model, model_custom)
        if not resolved:
            raise RuntimeError(
                "沒有選擇模型。\n"
                "→ 請在節點的 model 下拉挑一個（若清單是空的，先確認服務已啟動），"
                "或用 model_custom 直接輸入模型名稱。\n"
                f"→ 目前 provider：{provider}（{settings.get('base_url', '')}）"
            )
        return resolved

    @staticmethod
    def _encode_images(
        image_inputs: Dict[str, Any],
        *,
        task: Any,
        fmt: str,
        max_side: int,
        quality: int,
    ) -> Dict[str, providers_module.ImagePart]:
        """Encode every connected socket, in canonical IMAGE1..6 order."""
        encoded: Dict[str, providers_module.ImagePart] = {}
        for slot in IMAGE_SLOTS:
            tensor = image_inputs.get(slot)
            if tensor is None:
                continue
            try:
                image = imaging.encode_image(tensor, fmt=fmt, quality=quality, max_side=max_side)
            except Exception as error:
                raise RuntimeError(
                    f"{slot} 圖片編碼失敗：{error}\n→ 請確認上游節點輸出的是標準 IMAGE（0–1 的 float 張量）。"
                ) from None
            role = task.role_for(slot) if hasattr(task, "role_for") else ""
            encoded[slot] = providers_module.ImagePart(
                slot=slot,
                label=payload_module.image_tag(slot, role),
                data_b64=image.data_b64,
                mime=image.mime,
                role=role,
                width=image.width,
                height=image.height,
                source_width=image.source_width,
                source_height=image.source_height,
                byte_count=image.byte_count,
            )
        return encoded

    @staticmethod
    def _build_meta(
        request: providers_module.ChatRequest,
        result: providers_module.ChatResult,
        selected: Any,
        built: payload_module.BuiltPayload,
        images: Dict[str, providers_module.ImagePart],
        *,
        extra_notes: List[str],
        curated_chars: int,
        debug: bool,
    ) -> Dict[str, Any]:
        meta: Dict[str, Any] = {
            "task": {"id": selected.id, "title": selected.title, "category": selected.category},
            "provider": request.provider,
            "endpoint": result.endpoint,
            "model": result.model or request.model,
            "duration_ms": round(result.duration_ms, 1),
            "usage": result.usage,
            "output_chars": curated_chars,
            "thinking_chars": len(result.thinking or ""),
            "images": [
                dict(image.summary(), role=selected.role_for(slot)) for slot, image in images.items()
            ],
            "image_roles": {slot: selected.role_for(slot) for slot in built.used_slots},
            "request_size": providers_module.payload_size_hint(request),
            "notes": extra_notes,
        }
        if built.missing_declared:
            meta["missing_declared_images"] = built.missing_declared
        if debug:
            meta["request_preview"] = providers_module.render_request_preview(request)
        return meta


NODE_CLASS_MAPPINGS: Dict[str, Any] = {NODE_CLASS: TaskDrivenLLM}
NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {
    NODE_CLASS: "Task-Driven LLM API（提示詞任務 · LLM API）",
}
