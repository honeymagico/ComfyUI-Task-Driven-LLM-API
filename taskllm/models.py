"""Model discovery for the node's model dropdown.

``INPUT_TYPES`` is called on every page load and every
``app.refreshComboInNodes()``, so the probe has to be cheap: results are
cached, the timeout is clamped, and any failure degrades to the last good
list instead of raising (a stopped Ollama must never break ComfyUI's
startup or slow the UI down).
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional

from . import config as config_module
from . import providers as providers_module

#: How long a probe result stays fresh.
CACHE_TTL_SECONDS = 90

#: Ceiling for the probe issued from ``INPUT_TYPES``.
COMBO_PROBE_TIMEOUT = 2

_lock = threading.Lock()
_cache: Dict[str, Dict[str, Any]] = {}


def _key(provider: str, settings: Dict[str, Any]) -> str:
    return f"{provider}|{settings.get('base_url', '')}|{settings.get('api_key', '')[:6]}"


def list_models_cached(
    provider: str,
    settings: Optional[Dict[str, Any]] = None,
    *,
    timeout: Optional[int] = None,
    force: bool = False,
    raise_errors: bool = False,
) -> List[str]:
    """Return the provider's model names, using a short-lived cache.

    ``raise_errors`` is for the settings tab, which wants to show *why* a
    probe failed; the node's dropdown leaves it off so a stopped server can
    never break ``INPUT_TYPES``.
    """
    settings = dict(settings or config_module.provider_settings(provider))
    if timeout is not None:
        settings["timeout"] = max(1, min(int(timeout), int(settings.get("timeout") or 20)))
    key = _key(provider, settings)

    now = time.time()
    with _lock:
        entry = _cache.get(key)
    if entry and not force and now - entry["at"] < CACHE_TTL_SECONDS:
        return list(entry["models"])

    error: Optional[BaseException] = None
    try:
        models = providers_module.list_models(provider, settings)
    except Exception as failure:
        models = []
        error = failure

    with _lock:
        previous = _cache.get(key)
        if not models and previous:
            # Keep showing the last known-good list rather than emptying the
            # dropdown just because the server blipped.
            previous["at"] = now
            result = list(previous["models"])
        else:
            _cache[key] = {"at": now, "models": list(models)}
            result = list(models)

    if error is not None and raise_errors:
        raise error
    return result


def peek(provider: str, settings: Optional[Dict[str, Any]] = None) -> List[str]:
    """The cached list, without probing; empty when nothing was ever probed."""
    settings = dict(settings or config_module.provider_settings(provider))
    with _lock:
        entry = _cache.get(_key(provider, settings))
    return list(entry["models"]) if entry else []


def invalidate(provider: Optional[str] = None) -> None:
    with _lock:
        if provider is None:
            _cache.clear()
        else:
            for key in [key for key in _cache if key.startswith(provider + "|")]:
                _cache.pop(key, None)


def combo_choices(provider: str, fallback: str = "") -> List[str]:
    """Dropdown entries for ``INPUT_TYPES``; never raises, never blocks long."""
    settings = config_module.provider_settings(provider)
    models = list_models_cached(provider, settings, timeout=COMBO_PROBE_TIMEOUT)

    ordered: List[str] = []
    for candidate in (settings.get("default_model"), fallback, *models):
        if isinstance(candidate, str) and candidate.strip() and candidate not in ordered:
            ordered.append(candidate)
    if not ordered:
        ordered = ["（未偵測到模型：請啟動服務，或改用 model_custom 自行輸入）"]
    return ordered


def resolve_model(provider: str, settings: Dict[str, Any], model: str, model_custom: str = "") -> str:
    """Model precedence: ``model_custom`` > dropdown > provider default."""
    custom = (model_custom or "").strip()
    if custom:
        return custom
    chosen = (model or "").strip()
    if chosen and not chosen.startswith("（"):
        return chosen
    default_model = str(settings.get("default_model") or "").strip()
    if default_model:
        return default_model
    return ""
