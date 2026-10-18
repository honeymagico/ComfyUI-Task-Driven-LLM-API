"""Shared fixtures: a sandboxed data directory and a stub LLM server.

Import bootstrap
----------------
ComfyUI loads this folder as a package named after the directory, i.e.
``ComfyUI-Task-Driven-LLM-API``.  That name contains dashes, which is fine
for ComfyUI's path-based loader but *not* a valid Python identifier -- and
pytest's ``resolve_package_path`` refuses to treat a directory whose name
is not an identifier as a package.  Left alone, pytest therefore falls back
to importing the top-level ``__init__.py`` under the bare name ``__init__``,
where the package's relative imports blow up.

So the bootstrap below does two things, much like the sibling HonPrompt
repository: it loads the repository root as a package under a valid alias,
and primes ``sys.modules["__init__"]`` with that module so pytest's bogus
import becomes a harmless cache hit.  The node is then exercised through
exactly the same module objects ComfyUI would use.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List

import pytest

ROOT = Path(__file__).resolve().parent.parent

#: What ComfyUI calls the package (used only for documentation/assertions).
PACKAGE_NAME = ROOT.name

#: A valid identifier the tests can import under.
IMPORT_NAME = "task_driven_llm_api_node"

if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))


def _load_package() -> Any:
    existing = sys.modules.get(IMPORT_NAME)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(
        IMPORT_NAME,
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[IMPORT_NAME] = module
    spec.loader.exec_module(module)
    return module


package = _load_package()
# pytest will try ``importlib.import_module("__init__")`` for the repository
# root; a cache hit returns this module instead of re-executing the file.
sys.modules.setdefault("__init__", package)

# ``taskllm/__init__.py`` imports every internal module, so they are reachable
# as attributes and are guaranteed to be the very same module objects the
# node uses at execution time.
internals = importlib.import_module(IMPORT_NAME + ".taskllm")

paths = internals.paths
library_module = internals.library
config_module = internals.config
providers_module = internals.providers
models_module = internals.models
payload_module = internals.payload
imaging_module = internals.imaging
textutils_module = internals.textutils
nodes_module = importlib.import_module(IMPORT_NAME + ".nodes")


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Redirect every persisted file into ``tmp_path``.

    ``paths`` caches resolved directories, so the cache is cleared on the
    way in and on the way out.
    """
    monkeypatch.setattr(paths, "writable_root", lambda: tmp_path)
    paths._resolved.clear()  # noqa: SLF001 - test-only reset
    yield tmp_path
    paths._resolved.clear()  # noqa: SLF001


@pytest.fixture
def seeded_library(sandbox):
    """The shipped example library, copied into the sandbox."""
    source = ROOT / "prompts" / "tasks.default.json"
    target = sandbox / "prompts"
    target.mkdir(parents=True, exist_ok=True)
    (target / "tasks.default.json").write_bytes(source.read_bytes())
    library, warnings = library_module.load_library()
    assert not warnings
    return library


@pytest.fixture
def image_tensor():
    """Factory for a ``[1, H, W, 3]`` float tensor, as ComfyUI would pass."""
    torch = pytest.importorskip("torch")

    def make(height: int = 64, width: int = 48, value: float = 0.5):
        return torch.full((1, height, width, 3), value, dtype=torch.float32)

    return make


# --------------------------------------------------------------------------
# Stub LLM server
# --------------------------------------------------------------------------


class _StubState:
    def __init__(self) -> None:
        self.requests: List[Dict[str, Any]] = []
        self.ollama_response: Dict[str, Any] | None = None
        self.openai_response: Dict[str, Any] | None = None
        self.fail_status: int | None = None
        self.fail_body: Dict[str, Any] | None = None
        self.fail_next_think_unknown_field = False

    def last(self, path_contains: str = "") -> Dict[str, Any]:
        for entry in reversed(self.requests):
            if path_contains in entry["path"]:
                return entry
        raise AssertionError(f"沒有收到包含 {path_contains!r} 的請求：{[r['path'] for r in self.requests]}")

    def bodies(self, path_contains: str = "") -> List[Dict[str, Any]]:
        return [entry["body"] for entry in self.requests if path_contains in entry["path"]]

    def reset(self) -> None:
        self.requests.clear()
        self.fail_status = None
        self.fail_body = None


def _ollama_reply(body: Dict[str, Any]) -> Dict[str, Any]:
    messages = body.get("messages") or []
    images = [image for message in messages for image in (message.get("images") or [])]
    return {
        "model": body.get("model", "stub-model"),
        "message": {
            "role": "assistant",
            "content": f"STUB_OLLAMA images={len(images)} messages={len(messages)}",
        },
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 11,
        "eval_count": 22,
        "total_duration": 1_000_000,
    }


def _openai_reply(body: Dict[str, Any]) -> Dict[str, Any]:
    messages = body.get("messages") or []
    images = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            images += sum(1 for part in content if part.get("type") == "image_url")
    return {
        "model": body.get("model", "stub-model"),
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": f"STUB_OPENAI images={images}"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11},
    }


class _Handler(BaseHTTPRequestHandler):
    server_version = "StubLLM/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args: Any) -> None:  # keep pytest output clean
        return

    def _send(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path.startswith("/api/version"):
            self._send(200, {"version": "0.0-stub"})
        elif self.path.startswith("/api/tags"):
            self._send(200, {"models": [{"name": "stub-vl:latest"}, {"name": "stub-text:latest"}]})
        elif "/models" in self.path:
            self._send(200, {"data": [{"id": "stub-openai-model"}]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except ValueError:
            body = {}
        state: _StubState = self.server.state  # type: ignore[attr-defined]
        state.requests.append({"path": self.path, "body": body, "headers": dict(self.headers)})

        if state.fail_next_think_unknown_field and "think" in body:
            state.fail_next_think_unknown_field = False
            self._send(400, {"error": 'json: unknown field "think"'})
            return
        if state.fail_status:
            self._send(state.fail_status, state.fail_body or {"error": "boom"})
            return

        if self.path.startswith("/api/chat"):
            self._send(200, state.ollama_response or _ollama_reply(body))
        elif self.path.startswith("/v1/chat/completions"):
            self._send(200, state.openai_response or _openai_reply(body))
        else:
            self._send(404, {"error": "unknown path"})


class StubServer:
    def __init__(self, server: ThreadingHTTPServer) -> None:
        self._server = server
        self.state = server.state  # type: ignore[attr-defined]

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://127.0.0.1:{port}"

    @property
    def openai_url(self) -> str:
        return self.url + "/v1"

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def stub_llm():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.state = _StubState()  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    stub = StubServer(server)
    try:
        yield stub
    finally:
        stub.stop()
