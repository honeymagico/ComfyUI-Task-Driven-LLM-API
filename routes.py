"""HTTP endpoints behind the in-canvas prompt-library editor.

The node itself only ever needs the library at execution time; these
routes exist so the browser can read and write the same JSON without the
user leaving ComfyUI.  Every handler answers ``{"ok": bool, ...}`` and
never raises into aiohttp, so a mistake in the editor shows a readable
message instead of a stack trace.

Routes (all under ``/task_llm``):

===========================  ==================================================
``GET  /library``            full library + warnings + where it lives
``POST /library``            save the library (lenient, never loses work)
``POST /library/restore``    restore the shipped example library
``GET  /tasks``              dropdown labels only (used to refresh the combo)
``GET  /config``             provider settings, API keys masked
``POST /config``             save provider settings
``POST /models``             model list for a provider (force refresh)
``POST /probe``              connection test with a friendly diagnosis
``POST /preview``            render the exact messages for a wiring
``POST /open_folder``        reveal the library folder / config file
``GET  /export``             download tasks.json
``==========================  ==================================================
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
from typing import Any, Dict, Optional

from .taskllm import config as config_module
from .taskllm import library as library_module
from .taskllm import models as models_module
from .taskllm import paths
from .taskllm import payload as payload_module
from .taskllm import providers as providers_module
from .taskllm.providers import ImagePart

logger = logging.getLogger("task_driven_llm.routes")

ROUTE_PREFIX = "/task_llm"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _error(message: str, status: int = 400) -> Any:
    from aiohttp import web  # noqa: PLC0415

    return web.json_response({"ok": False, "error": message}, status=status)


async def _json_body(request: Any) -> Any:
    try:
        return await request.json()
    except Exception:
        return {}


def _library_payload(include_tasks: bool = True) -> Dict[str, Any]:
    library, warnings = library_module.load_library()
    data: Dict[str, Any] = {
        "ok": True,
        "library": library.to_dict(),
        "warnings": warnings,
        "labels": library.labels(),
        "locate": paths.locate(),
    }
    if include_tasks:
        data["tasks"] = [
            {"id": task.id, "label": task.label, "enabled": task.enabled} for task in library.tasks
        ]
    return data


def _open_path(target: str) -> str:
    """Best-effort "reveal in file manager"; the path is returned regardless.

    ComfyUI often runs as a service account with no desktop session, so the
    launch is fired on a thread and its failure is deliberately ignored --
    the UI always shows the path with a copy button as the real fallback.
    """
    if sys.platform.startswith("win"):
        command = ["explorer", target]
    elif sys.platform == "darwin":
        command = ["open", target]
    else:
        command = ["xdg-open", target]

    def _spawn() -> None:
        try:
            subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        except Exception as error:  # pragma: no cover - environment dependent
            logger.info("無法自動開啟 %s：%s", target, error)

    threading.Thread(target=_spawn, daemon=True).start()
    return target


def _placeholder_images(slots: list, task: Any = None) -> Dict[str, ImagePart]:
    """Fake images with realistic metadata, for the editor's preview pane."""
    images: Dict[str, ImagePart] = {}
    for slot in slots:
        if slot not in library_module.IMAGE_SLOTS:
            continue
        role = task.role_for(slot) if task is not None and hasattr(task, "role_for") else ""
        images[slot] = ImagePart(
            slot=slot,
            label=payload_module.image_tag(slot, role),
            data_b64="",
            mime="image/jpeg",
            role=role,
            width=1024,
            height=1024,
            byte_count=180 * 1024,
        )
    return images


def _task_from_body(task_body: Dict[str, Any]) -> Any:
    """Validate one task dict exactly the way the library parser would."""
    library, warnings = library_module.coerce_library({"tasks": [task_body]})
    if not library.tasks:
        raise ValueError(warnings[0] if warnings else "這個 TASK 沒有標題也沒有內容。")
    return library.tasks[0], warnings


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------


def register_routes(router: Optional[Any] = None) -> bool:
    """Attach every route to ComfyUI's aiohttp router.

    Returns ``False`` when ComfyUI is not present (unit tests, plain
    scripts) instead of failing the import.
    """
    if router is None:
        try:
            from server import PromptServer  # noqa: PLC0415
        except Exception:
            logger.debug("找不到 ComfyUI 的 PromptServer，略過 API 路由註冊。")
            return False
        instance = getattr(PromptServer, "instance", None)
        if instance is None:
            return False
        router = instance.routes

    from aiohttp import web  # noqa: PLC0415

    # -- prompt library ----------------------------------------------------

    @router.get(f"{ROUTE_PREFIX}/library")
    async def get_library(_request: Any) -> Any:
        try:
            return web.json_response(_library_payload())
        except Exception as error:
            logger.exception("讀取提示詞庫失敗")
            return _error(f"讀取提示詞庫失敗：{error}", 500)

    @router.post(f"{ROUTE_PREFIX}/library")
    async def save_library(request: Any) -> Any:
        body = await _json_body(request)
        raw = body.get("library") if isinstance(body, dict) else None
        if raw is None:
            raw = body
        try:
            library, warnings = library_module.coerce_library(raw)
            path = library_module.save_library(library)
        except Exception as error:
            return _error(f"提示詞庫儲存失敗：{error}")
        return web.json_response(
            {
                "ok": True,
                "warnings": warnings,
                "labels": library.labels(),
                "path": str(path),
                "locate": paths.locate(),
            }
        )

    @router.post(f"{ROUTE_PREFIX}/library/restore")
    async def restore_library(_request: Any) -> Any:
        seed = paths.default_tasks_path()
        if not seed.exists():
            return _error(f"找不到內建範例檔案：{seed}")
        try:
            library, warnings = library_module._read(seed)  # noqa: SLF001 - same package
            library_module.save_library(library)
        except Exception as error:
            return _error(f"還原失敗：{error}")
        return web.json_response(
            {"ok": True, "warnings": warnings, "library": library.to_dict(), "labels": library.labels()}
        )

    @router.get(f"{ROUTE_PREFIX}/tasks")
    async def get_tasks(_request: Any) -> Any:
        try:
            return web.json_response({"ok": True, "labels": library_module.task_labels()})
        except Exception as error:
            return _error(f"讀取 TASK 清單失敗：{error}", 500)

    @router.get(f"{ROUTE_PREFIX}/export")
    async def export_library(_request: Any) -> Any:
        try:
            library, _warnings = library_module.load_library()
            body = json.dumps(library.to_dict(), ensure_ascii=False, indent=2)
        except Exception as error:
            return _error(f"匯出失敗：{error}", 500)
        return web.Response(
            body=body.encode("utf-8"),
            content_type="application/json",
            headers={"Content-Disposition": 'attachment; filename="tasks.json"'},
        )

    @router.post(f"{ROUTE_PREFIX}/import")
    async def import_library(request: Any) -> Any:
        body = await _json_body(request)
        raw = body.get("library") if isinstance(body, dict) else None
        if raw is None:
            raw = body
        try:
            library, warnings = library_module.coerce_library(raw)
            library_module.save_library(library)
        except Exception as error:
            return _error(f"匯入失敗：{error}")
        return web.json_response(
            {"ok": True, "warnings": warnings, "library": library.to_dict(), "labels": library.labels()}
        )

    # -- provider settings -------------------------------------------------

    @router.get(f"{ROUTE_PREFIX}/config")
    async def get_config(_request: Any) -> Any:
        try:
            return web.json_response({"ok": True, "config": config_module.public_config()})
        except Exception as error:
            return _error(f"讀取設定失敗：{error}", 500)

    @router.post(f"{ROUTE_PREFIX}/config")
    async def save_config(request: Any) -> Any:
        body = await _json_body(request)
        raw = body.get("config") if isinstance(body, dict) and "config" in body else body
        try:
            config_module.apply_public_update(raw)
            models_module.invalidate()
        except Exception as error:
            return _error(f"設定儲存失敗：{error}")
        return web.json_response({"ok": True, "config": config_module.public_config()})

    # -- discovery / diagnostics -------------------------------------------

    @router.post(f"{ROUTE_PREFIX}/models")
    async def list_models(request: Any) -> Any:
        body = await _json_body(request)
        provider = str((body or {}).get("provider") or config_module.PROVIDER_OLLAMA)
        if provider not in config_module.PROVIDER_IDS:
            return _error(f"不認識的 provider：{provider}")
        settings = config_module.provider_settings(provider)
        overrides = (body or {}).get("settings")
        if isinstance(overrides, dict):
            settings.update({key: value for key, value in overrides.items() if value not in (None, "")})
        try:
            models = models_module.list_models_cached(
                provider,
                settings,
                timeout=int(settings.get("timeout") or 20),
                force=True,
                raise_errors=True,
            )
        except providers_module.ProviderError as error:
            return web.json_response(
                {
                    "ok": False,
                    "error": str(error),
                    "hint": error.hint,
                    # Stale names beat an empty dropdown while the server is down.
                    "models": models_module.peek(provider, settings),
                }
            )
        except Exception as error:
            return web.json_response({"ok": False, "error": f"取得模型清單失敗：{error}", "models": []})
        return web.json_response({"ok": True, "provider": provider, "models": models, "settings": settings})

    @router.post(f"{ROUTE_PREFIX}/probe")
    async def probe(request: Any) -> Any:
        body = await _json_body(request)
        provider = str((body or {}).get("provider") or config_module.PROVIDER_OLLAMA)
        if provider not in config_module.PROVIDER_IDS:
            return _error(f"不認識的 provider：{provider}")
        settings = config_module.provider_settings(provider)
        overrides = (body or {}).get("settings")
        if isinstance(overrides, dict):
            settings.update({key: value for key, value in overrides.items() if value not in (None, "")})
        try:
            result = providers_module.probe(provider, settings)
        except providers_module.ProviderError as error:
            return web.json_response({"ok": False, "error": str(error), "hint": error.hint})
        except Exception as error:
            return web.json_response({"ok": False, "error": f"連線測試失敗：{error}"})
        return web.json_response({"ok": True, **result})

    # -- payload preview ---------------------------------------------------

    @router.post(f"{ROUTE_PREFIX}/preview")
    async def preview(request: Any) -> Any:
        body = await _json_body(request) or {}
        library, _warnings = library_module.load_library()

        # The editor sends the *unsaved* task it is showing so the preview is
        # never stale; that single body is validated through the same parser
        # the library file goes through.
        task_body = body.get("task_body")
        if isinstance(task_body, dict):
            default_system = str(body.get("default_system") or library.default_system)
            library = library_module.Library(default_system=default_system)
            try:
                task, _task_warnings = _task_from_body(task_body)
            except Exception as error:
                return _error(f"預覽失敗：{error}")
        else:
            task = library.resolve(str(body.get("task") or ""))
            if task is None:
                return _error("找不到指定的 TASK，請先存檔並重新載入清單。")

        connected = body.get("connected") or []
        if not isinstance(connected, list):
            connected = []
        images = _placeholder_images([str(slot).upper() for slot in connected], task)

        built = payload_module.build_payload(
            task,
            library,
            images,
            user_prompt=str(body.get("user_prompt") or ""),
            system_override=str(body.get("system_override") or ""),
            delivery=str(body.get("image_delivery") or payload_module.DELIVERY_PER_IMAGE),
        )
        request_obj = built.to_chat_request(
            provider=config_module.PROVIDER_OLLAMA,
            base_url="",
            model="(preview)",
        )
        return web.json_response(
            {
                "ok": True,
                "transcript": providers_module.render_request_preview(request_obj),
                "system": built.system,
                "messages": [turn.summary() for turn in built.turns],
                "notes": built.notes,
                "used_slots": built.used_slots,
                "missing_declared": built.missing_declared,
                "request_size": providers_module.payload_size_hint(request_obj),
            }
        )

    # -- filesystem --------------------------------------------------------

    @router.post(f"{ROUTE_PREFIX}/open_folder")
    async def open_folder(request: Any) -> Any:
        body = await _json_body(request)
        target = str((body or {}).get("target") or "prompts")
        location = paths.locate()
        resolved = {
            "prompts": location["prompts_dir"],
            "tasks": location["tasks_path"],
            "default": location["default_tasks_path"],
            "config": location["config_path"],
            "node": location["node_dir"],
        }.get(target, location["prompts_dir"])
        _open_path(resolved)
        return web.json_response({"ok": True, "path": resolved, "target": target, "locate": location})

    logger.info("Task-Driven LLM API：已註冊 %s 的路由。", ROUTE_PREFIX)
    return True
