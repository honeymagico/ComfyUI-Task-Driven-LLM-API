"""Task-Driven LLM API -- ComfyUI custom node package.

ComfyUI discovers this folder because it exposes ``NODE_CLASS_MAPPINGS``
here.  The package is one node plus an in-canvas prompt-library editor:

* ``nodes.py``   the ``TaskDrivenLLM`` node (TASK + images -> LLM API -> text)
* ``routes.py``  the ``/task_llm/*`` HTTP endpoints the editor talks to
* ``taskllm/``   framework-free internals (library, payload, providers, imaging)
* ``web/``       the browser extension: editor panel, buttons, live preview
* ``prompts/``   ``tasks.json`` (yours) and ``tasks.default.json`` (examples)

See README.md for installation and the TASK authoring guide.
"""

import logging

from . import routes
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, TaskDrivenLLM

logger = logging.getLogger("task_driven_llm")

#: Served at /extensions/<this folder>/ -- the editor lives there.
WEB_DIRECTORY = "./web"

try:
    # Attaches /task_llm/* to ComfyUI's aiohttp router.  Safe to call
    # outside ComfyUI (returns False), which is what unit tests rely on.
    routes.register_routes()
except Exception:  # pragma: no cover - must never stop ComfyUI from booting
    logger.exception("Task-Driven LLM API：API 路由註冊失敗，節點仍可使用，但編輯器無法存取提示詞庫。")

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
    "TaskDrivenLLM",
]
