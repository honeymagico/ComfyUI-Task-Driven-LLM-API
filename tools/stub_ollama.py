#!/usr/bin/env python3
"""A tiny fake Ollama for testing the Task-Driven LLM API node offline.

It speaks just enough of the native Ollama API (``/api/version``,
``/api/tags``, ``/api/chat``) for the node's dropdown, connection test and
execution to all succeed, and it *echoes back what it received* instead of
generating text.  That makes the image-tagging scheme directly visible: the
node prints the reply on its canvas, so you can see whether each picture
arrived with the right ``【IMAGEn】`` marker.

Usage::

    python tools/stub_ollama.py                # listens on 127.0.0.1:11434
    python tools/stub_ollama.py --port 11500

Then point the node at it (or let it sit on the default port) and queue a
prompt.  No models are downloaded and no GPU is used.

Options:
    --host HOST      bind address (default 127.0.0.1)
    --port PORT      bind port (default 11434)
    --models A,B     model names to advertise (default: stub-vl:8b,stub-text:8b)
    --delay SECONDS  pretend to think for a while, to exercise timeouts
    --verbose        log every request body
"""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_MODELS = ["stub-vl:8b", "stub-text:8b"]


class StubState:
    def __init__(self, models: list[str], delay: float, verbose: bool) -> None:
        self.models = models
        self.delay = delay
        self.verbose = verbose


def _content_text(message: dict) -> str:
    """Both wire formats: Ollama sends a string, OpenAI sends content parts."""
    content = message.get("content")
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content or ""


def _image_count(message: dict) -> int:
    """Ollama carries ``images``; OpenAI carries ``image_url`` parts."""
    images = message.get("images")
    if isinstance(images, list) and images:
        return len(images)
    content = message.get("content")
    if isinstance(content, list):
        return sum(1 for part in content if isinstance(part, dict) and part.get("type") == "image_url")
    return 0


def describe(body: dict) -> str:
    """Summarise the request so the mapping is readable on the node canvas."""
    messages = body.get("messages") or []
    system = [m for m in messages if m.get("role") == "system"]
    turns = [m for m in messages if m.get("role") != "system"]

    tagged: list[str] = []
    untagged_with_images: list[str] = []
    image_count = 0
    for message in turns:
        images = _image_count(message)
        image_count += images
        text = _content_text(message)
        label = text.splitlines()[0][:60] if text else ""
        if not images:
            continue
        if "【IMAGE" in text:
            tagged.append(f"{label}（{images} 張）")
        else:
            untagged_with_images.append(label or "（沒有文字）")

    lines = ["【Stub Ollama】我收到了以下內容：", ""]
    lines.append(f"系統提示詞：{len(_content_text(system[0])) if system else 0} 字")
    lines.append(f"訊息數：{len(turns)}　圖片數：{image_count}")
    lines.append("")
    if tagged:
        lines.append("帶著接口標記送來的圖片：")
        lines.extend(f"  · {entry}" for entry in tagged)
    if untagged_with_images:
        lines.append("")
        lines.append("有圖片但內容裡沒有 【IMAGEn】 標記：")
        lines.extend(f"  · {entry}" for entry in untagged_with_images)
        lines.append("  （「集中單一訊息」模式本來就沒有逐張標記，屬正常；")
        lines.append("    「逐張標註」模式下這代表標記沒有送出，請回報。）")
    if not image_count:
        lines.append("這次沒有任何圖片。")

    option_keys = sorted((body.get("options") or {}).keys())
    lines.append("")
    lines.append(
        "參數：model={} think={} keep_alive={} options={}".format(
            body.get("model"), body.get("think"), body.get("keep_alive"), ",".join(option_keys) or "無"
        )
    )
    instruction = next(
        (_content_text(m) for m in reversed(turns) if not _image_count(m) and _content_text(m)),
        "",
    )
    if instruction:
        lines.append("")
        lines.append("最後一則指示開頭：")
        lines.append("  " + instruction.replace("\n", " ")[:160])
    return "\n".join(lines)


class Handler(BaseHTTPRequestHandler):
    server_version = "StubOllama/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:  # keep the console readable
        if self.server.state.verbose:  # type: ignore[attr-defined]
            super().log_message(fmt, *args)

    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        state: StubState = self.server.state  # type: ignore[attr-defined]
        if self.path.startswith("/api/version"):
            self._send(200, {"version": "0.0-stub"})
        elif self.path.startswith("/api/tags"):
            self._send(200, {"models": [{"name": name, "model": name, "size": 1} for name in state.models]})
        elif self.path.startswith("/api/ps"):
            self._send(200, {"models": []})
        elif "/models" in self.path:  # OpenAI-compatible listing
            self._send(200, {"data": [{"id": name} for name in state.models]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        state: StubState = self.server.state  # type: ignore[attr-defined]
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except ValueError:
            self._send(400, {"error": "invalid json"})
            return

        if state.verbose:
            print(f"[stub-ollama] {self.path}: {json.dumps(body, ensure_ascii=False)[:600]}", flush=True)
        if state.delay:
            time.sleep(state.delay)

        if self.path.startswith("/api/chat"):
            reply = describe(body)
            self._send(
                200,
                {
                    "model": body.get("model", "stub"),
                    "created_at": "2026-01-01T00:00:00Z",
                    "message": {"role": "assistant", "content": reply},
                    "done": True,
                    "done_reason": "stop",
                    "prompt_eval_count": 1,
                    "eval_count": len(reply),
                },
            )
        elif self.path.startswith("/v1/chat/completions"):
            self._send(
                200,
                {
                    "model": body.get("model", "stub"),
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": describe(body)}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            )
        else:
            self._send(404, {"error": "unknown path"})


def main() -> None:
    parser = argparse.ArgumentParser(description="Fake Ollama server for offline node testing.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11434)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    models = [name.strip() for name in args.models.split(",") if name.strip()]
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.state = StubState(models, args.delay, args.verbose)  # type: ignore[attr-defined]
    print(f"Stub Ollama listening on http://{args.host}:{args.port} (models: {', '.join(models)})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
