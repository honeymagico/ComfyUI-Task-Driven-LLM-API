"""Task-Driven LLM API -- internals for the ComfyUI custom node.

The public surface of this package is the node class in ``nodes.py`` and
the HTTP routes in ``routes.py``.  Everything here is deliberately free
of ComfyUI imports so the prompt library, the payload builder and the
provider adapters can be unit-tested (and reused from a plain script)
without a running ComfyUI.

Module map
----------
``paths``      where the prompt library and config.json live (with a
               read-only-filesystem fallback).
``config``     provider settings: base URL, API key, defaults, env vars.
``library``    the TASK prompt library: schema, validation, save/load.
``payload``    turns the selected TASK + connected images into the exact
               message list an LLM receives (the IMAGE1..n convention).
``providers``  one adapter per LLM API (Ollama native, OpenAI-compatible).
``imaging``    torch tensor -> resized JPEG/PNG base64.
``models``     cached model discovery for the node's model dropdown.
``textutils``  output cleanup (reasoning blocks, code fences).
"""

from . import config, imaging, library, models, paths, payload, providers, textutils

__all__ = ["config", "imaging", "library", "models", "paths", "payload", "providers", "textutils"]
