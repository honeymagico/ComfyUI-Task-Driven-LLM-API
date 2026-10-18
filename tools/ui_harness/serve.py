#!/usr/bin/env python3
"""Render the prompt-library editor in a plain browser, without ComfyUI.

Why this exists
---------------
The editor is a lot of DOM that only ever appears after a click, so the usual
way to check it is to open ComfyUI, add a node and click a button.  This
serves the *real* ``web/task_llm.js`` and ``web/task_llm.css`` against a
stand-in ComfyUI host (``scripts/app.js`` + ``scripts/api.js``) so any change
to the panel can be inspected -- and screenshotted -- in one command.

The mocked API responses are generated from the real Python modules
(:mod:`taskllm.library`, :mod:`taskllm.payload`, :mod:`taskllm.config`), so
the data on screen is the data the node would really send.  The mock is
regenerated on every start; it cannot drift.

Usage::

    python tools/ui_harness/serve.py            # http://127.0.0.1:8190
    python tools/ui_harness/serve.py --port 8200

    # Screenshot with headless Firefox (the slow image in the page holds the
    # load event open until the panel has finished rendering):
    firefox --headless --window-size=1440,940 \\
        --screenshot docs/editor-tasks.png \\
        "http://127.0.0.1:8190/index.html?theme=dark"

Query parameters understood by ``index.html``:

``theme=dark|light``  pick the host theme tokens to emulate
``tab=config``        open on the connection-settings tab
``scroll=bottom``     scroll the form to the bottom (the preview section)
``mode=empty``        pretend the library has no TASKs
``diag=1``            print DOM counts into the page (verification aid)
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

HARNESS_DIR = Path(__file__).resolve().parent
REPO_DIR = HARNESS_DIR.parent.parent
sys.path.insert(0, str(REPO_DIR))

# 1x1 transparent PNG: served slowly, to hold the load event open.
SLOW_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d494844520000000100000001080600000"
    "01f15c4890000000a49444154789c6360000002000100ffff0300000600"
    "05570d5d0000000049454e44ae426082"
)


def build_mock() -> dict:
    """Generate the mocked API payloads from the real modules."""
    from taskllm import config as config_module
    from taskllm import library as library_module
    from taskllm import payload as payload_module
    from taskllm import providers as providers_module

    seed = REPO_DIR / "prompts" / "tasks.default.json"
    raw = json.loads(seed.read_text(encoding="utf-8"))
    library, warnings = library_module.coerce_library(raw)
    task = library.resolve("提示詞生成 · 多圖參考合成（景 / 人 / 色調）") or library.tasks[0]

    def preview(task_body: dict, connected: list, user_prompt: str = "") -> dict:
        one, _ = library_module.coerce_library({"tasks": [task_body]})
        chosen = one.tasks[0]
        images = {}
        for slot in connected:
            role = chosen.role_for(slot)
            images[slot] = providers_module.ImagePart(
                slot=slot,
                label=payload_module.image_tag(slot, role),
                data_b64="",
                mime="image/jpeg",
                role=role,
                width=1024,
                height=768,
                byte_count=180 * 1024,
            )
        built = payload_module.build_payload(
            chosen,
            library_module.Library(default_system=library.default_system),
            images,
            user_prompt=user_prompt,
            delivery=payload_module.DELIVERY_PER_IMAGE,
        )
        request = built.to_chat_request(provider="ollama", base_url="", model="(preview)")
        return {
            "ok": True,
            "transcript": providers_module.render_request_preview(request),
            "system": built.system,
            "messages": [turn.summary() for turn in built.turns],
            "notes": built.notes,
            "used_slots": built.used_slots,
            "missing_declared": built.missing_declared,
            "request_size": providers_module.payload_size_hint(request),
        }

    public = config_module.public_config()
    node_dir = "/ComfyUI/custom_nodes/ComfyUI-Task-Driven-LLM-API"
    return {
        "library": {
            "ok": True,
            "library": library.to_dict(),
            "warnings": warnings,
            "labels": library.labels(),
            "locate": {
                "node_dir": node_dir,
                "data_root": node_dir,
                "prompts_dir": f"{node_dir}/prompts",
                "tasks_path": f"{node_dir}/prompts/tasks.json",
                "default_tasks_path": f"{node_dir}/prompts/tasks.default.json",
                "config_path": f"{node_dir}/config.json",
                "node_dir_writable": True,
                "prompts_dir_writable": True,
                "using_fallback": False,
            },
        },
        "config": {"ok": True, "config": public},
        "preview_default": preview(task.to_dict(), ["IMAGE1", "IMAGE2"], user_prompt="改成夜晚的市場"),
        "models": {
            "ok": True,
            "provider": "ollama",
            "models": ["qwen3-vl:8b", "gemma3:12b", "minicpm-v:8b", "llava:13b"],
        },
        "probe": {
            "ok": True,
            "provider": "ollama",
            "endpoint": "http://127.0.0.1:11434",
            "version": "0.5.4",
            "models": ["qwen3-vl:8b", "gemma3:12b"],
            "message": "Ollama 0.5.4 連線正常，找到 2 個模型。",
        },
    }


def write_mock() -> Path:
    target = HARNESS_DIR / "scripts" / "mock-data.js"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "// Generated by tools/ui_harness/serve.py from the real Python modules.\n"
        "export const MOCK = " + json.dumps(build_mock(), ensure_ascii=False, indent=2) + ";\n",
        encoding="utf-8",
    )
    return target


class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):  # keep the console readable
        if self.server.verbose:  # type: ignore[attr-defined]
            super().log_message(fmt, *args)

    def _send_file(self, path: Path, content_type: str) -> None:
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 - http.server API
        route = urlparse(self.path).path

        if route.startswith("/slow.png"):
            time.sleep(1.4)
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(SLOW_PNG)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(SLOW_PNG)
            return

        # Serve the real extension files, so the harness always tests live code.
        prefix = "/extensions/ComfyUI-Task-Driven-LLM-API/"
        if route.startswith(prefix):
            candidate = (REPO_DIR / "web" / route[len(prefix) :]).resolve()
            if candidate.is_file() and REPO_DIR in candidate.parents:
                kind = "text/css" if candidate.suffix == ".css" else "text/javascript"
                self._send_file(candidate, kind)
                return
            self.send_error(404)
            return

        if route == "/scripts/mock-data.js":
            mock = HARNESS_DIR / "scripts" / "mock-data.js"
            if not mock.exists():
                write_mock()
            self._send_file(mock, "text/javascript")
            return

        self.directory = str(HARNESS_DIR)
        super().do_GET()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8190)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    written = write_mock()
    print(f"mock data -> {written.relative_to(REPO_DIR)}")
    print(f"UI harness on http://{args.host}:{args.port}/index.html?theme=dark", flush=True)

    handler = functools.partial(Handler, directory=str(HARNESS_DIR))
    http.server.ThreadingHTTPServer.allow_reuse_address = True
    server = http.server.ThreadingHTTPServer((args.host, args.port), handler)
    server.verbose = args.verbose  # type: ignore[attr-defined]
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
