"""Provider configuration: base URLs, API keys and per-provider defaults.

Resolution order, lowest priority first:

1. the built-in defaults below;
2. ``config.json`` next to the node (written by the in-canvas settings tab);
3. environment variables (``TASKLLM_OLLAMA_BASE_URL``, ``OLLAMA_HOST``, …).

Local Ollama needs no API key at all -- it ignores the ``Authorization``
header -- so the shipped default is a placeholder value rather than an
empty string.  OpenAI-compatible endpoints such as OpenRouter or DeepSeek
do need one, which is why keys live in ``config.json`` (gitignored) and
never in the workflow, so a workflow can be shared safely.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import paths

#: Provider ids; these strings appear in workflows, so they must stay stable.
PROVIDER_OLLAMA = "ollama"
PROVIDER_OPENAI = "openai_compatible"
PROVIDER_IDS = (PROVIDER_OLLAMA, PROVIDER_OPENAI)

#: Everything the settings tab needs to render a provider card.
PROVIDER_INFO: Dict[str, Dict[str, Any]] = {
    PROVIDER_OLLAMA: {
        "label": "Ollama（本地）",
        "summary": "呼叫本機 Ollama 的 /api/chat，支援視覺模型（qwen3-vl、gemma3、minicpm-v…）。",
        "needs_key": False,
        "key_hint": "Ollama 不需要金鑰，留著預設的 ollama 即可。",
        "base_url_hint": "例如 http://127.0.0.1:11434",
        "supports_think": True,
        "supports_keep_alive": True,
    },
    PROVIDER_OPENAI: {
        "label": "OpenAI 相容端點",
        "summary": "任何 /v1/chat/completions 端點：LM Studio、llama.cpp server、vLLM、OpenRouter、DeepSeek、SiliconFlow…",
        "needs_key": True,
        "key_hint": "多數雲端服務需要金鑰；本機 LM Studio / llama.cpp 可留空。",
        "base_url_hint": "例如 http://127.0.0.1:1234/v1",
        "supports_think": False,
        "supports_keep_alive": False,
    },
}

#: One-click base URLs for the OpenAI-compatible adapter.
BASE_URL_PRESETS: List[Dict[str, str]] = [
    {"label": "LM Studio（本機）", "provider": PROVIDER_OPENAI, "base_url": "http://127.0.0.1:1234/v1"},
    {"label": "llama.cpp server（本機）", "provider": PROVIDER_OPENAI, "base_url": "http://127.0.0.1:8080/v1"},
    {"label": "vLLM（本機）", "provider": PROVIDER_OPENAI, "base_url": "http://127.0.0.1:8000/v1"},
    {"label": "OpenRouter", "provider": PROVIDER_OPENAI, "base_url": "https://openrouter.ai/api/v1"},
    {"label": "DeepSeek", "provider": PROVIDER_OPENAI, "base_url": "https://api.deepseek.com/v1"},
    {"label": "SiliconFlow", "provider": PROVIDER_OPENAI, "base_url": "https://api.siliconflow.cn/v1"},
    {"label": "OpenAI", "provider": PROVIDER_OPENAI, "base_url": "https://api.openai.com/v1"},
    {"label": "Ollama（OpenAI 相容介面）", "provider": PROVIDER_OPENAI, "base_url": "http://127.0.0.1:11434/v1"},
]

DEFAULT_CONFIG: Dict[str, Any] = {
    "version": 1,
    "default_provider": PROVIDER_OLLAMA,
    # Only encoding knobs that have no node widget live here; image_max_side
    # and image_format are per-node widgets, so keeping copies in config.json
    # would just be two sources of truth.
    "image_quality": 90,
    "providers": {
        PROVIDER_OLLAMA: {
            "base_url": "http://127.0.0.1:11434",
            "api_key": "ollama",
            "default_model": "",
            "keep_alive": "5m",
            "timeout": 300,
        },
        PROVIDER_OPENAI: {
            "base_url": "http://127.0.0.1:1234/v1",
            "api_key": "",
            "default_model": "",
            "timeout": 300,
        },
    },
}

#: Fields copied out of config.json / the settings tab per provider.
_PROVIDER_FIELDS = ("base_url", "api_key", "default_model", "keep_alive", "timeout")


def _env_prefix(provider: str) -> str:
    return "TASKLLM_" + "".join(ch if ch.isalnum() else "_" for ch in provider).upper()


def _deep_copy(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _deep_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_deep_copy(item) for item in value]
    return value


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Merge the shipped defaults with ``config.json`` (never raises)."""
    target = Path(path) if path is not None else paths.config_path()
    config = _deep_copy(DEFAULT_CONFIG)
    try:
        raw = paths.read_json(target)
    except FileNotFoundError:
        raw = None
    except Exception:
        raw = None

    if isinstance(raw, dict):
        if raw.get("default_provider") in PROVIDER_IDS:
            config["default_provider"] = raw["default_provider"]
        if raw.get("image_quality") is not None:
            config["image_quality"] = raw["image_quality"]
        incoming = raw.get("providers")
        if isinstance(incoming, dict):
            for provider, values in incoming.items():
                if provider not in PROVIDER_INFO or not isinstance(values, dict):
                    continue
                target_provider = config["providers"].setdefault(provider, {})
                for key in _PROVIDER_FIELDS:
                    if key in values:
                        target_provider[key] = values[key]
    return config


def save_config(config: Dict[str, Any], path: Optional[Path] = None) -> Path:
    target = Path(path) if path is not None else paths.config_path()
    paths.atomic_write_json(target, config)
    return target


def provider_settings(provider: str) -> Dict[str, Any]:
    """Effective settings for ``provider`` after defaults, file and env merge."""
    provider = provider if provider in PROVIDER_INFO else PROVIDER_OLLAMA
    config = load_config()
    settings = _deep_copy(DEFAULT_CONFIG["providers"].get(provider, {}))
    settings.update(
        {key: value for key, value in config["providers"].get(provider, {}).items() if value is not None}
    )

    prefix = _env_prefix(provider)
    for key in _PROVIDER_FIELDS:
        value = os.environ.get(f"{prefix}_{key.upper()}")
        if value not in (None, ""):
            settings[key] = value

    if provider == PROVIDER_OLLAMA:
        # Ollama's own client honours OLLAMA_HOST; respect it so the node
        # agrees with `ollama` CLI setups.
        host = os.environ.get("OLLAMA_HOST")
        if host:
            settings["base_url"] = host if "://" in host else f"http://{host}"
        if not settings.get("api_key"):
            settings["api_key"] = "ollama"

    settings["provider"] = provider
    settings["base_url"] = str(settings.get("base_url") or "").strip().rstrip("/")
    settings["api_key"] = str(settings.get("api_key") or "")
    settings["default_model"] = str(settings.get("default_model") or "").strip()
    try:
        settings["timeout"] = max(5, int(float(settings.get("timeout") or 300)))
    except (TypeError, ValueError):
        settings["timeout"] = 300
    if settings.get("keep_alive") is None:
        settings["keep_alive"] = ""
    return settings


def public_config() -> Dict[str, Any]:
    """Config for the settings tab: keys are replaced by a ``set`` flag.

    The API key never travels back to the browser, so saving from the UI
    cannot leak an existing key by accident.  Omitting ``api_key`` from a
    save payload means "keep the stored value"; sending an empty string
    means "clear it".
    """
    config = load_config()
    providers: Dict[str, Any] = {}
    for provider in PROVIDER_IDS:
        settings = config["providers"].get(provider, {})
        key = str(settings.get("api_key") or "")
        providers[provider] = {
            "base_url": settings.get("base_url", ""),
            "default_model": settings.get("default_model", ""),
            "keep_alive": settings.get("keep_alive", ""),
            "timeout": settings.get("timeout", 300),
            "api_key_set": bool(key),
            "api_key_hint": ("••••••" + key[-4:]) if len(key) > 4 else ("••••" if key else ""),
            "env_override": _env_override_fields(provider),
        }
    return {
        "version": config.get("version", 1),
        "default_provider": config.get("default_provider", PROVIDER_OLLAMA),
        "image_quality": config.get("image_quality", 90),
        "providers": providers,
        "provider_info": PROVIDER_INFO,
        "base_url_presets": BASE_URL_PRESETS,
        "path": str(paths.config_path()),
    }


def image_defaults() -> Dict[str, int]:
    """Encoding settings that have no node widget of their own."""
    try:
        quality = int(float(load_config().get("image_quality") or 90))
    except (TypeError, ValueError):
        quality = 90
    return {"quality": max(1, min(100, quality))}


def _env_override_fields(provider: str) -> List[str]:
    prefix = _env_prefix(provider)
    found = [key for key in _PROVIDER_FIELDS if os.environ.get(f"{prefix}_{key.upper()}")]
    if provider == PROVIDER_OLLAMA and os.environ.get("OLLAMA_HOST"):
        found.append("base_url")
    return sorted(set(found))


def apply_public_update(incoming: Any) -> Dict[str, Any]:
    """Merge a settings-tab payload into ``config.json`` and save it."""
    if not isinstance(incoming, dict):
        raise ValueError("設定內容必須是 JSON 物件。")

    config = load_config()
    if incoming.get("default_provider") in PROVIDER_IDS:
        config["default_provider"] = incoming["default_provider"]
    if incoming.get("image_quality") is not None:
        try:
            config["image_quality"] = max(1, min(100, int(float(incoming["image_quality"]))))
        except (TypeError, ValueError):
            pass

    providers = incoming.get("providers")
    if isinstance(providers, dict):
        for provider, values in providers.items():
            if provider not in PROVIDER_INFO or not isinstance(values, dict):
                continue
            target = config["providers"].setdefault(provider, {})
            for key in ("base_url", "default_model", "keep_alive"):
                if key in values and values[key] is not None:
                    target[key] = str(values[key]).strip()
            if "timeout" in values and values["timeout"] is not None:
                try:
                    target["timeout"] = max(5, int(float(values["timeout"])))
                except (TypeError, ValueError):
                    pass
            # Absent key == keep; empty string == clear.
            if "api_key" in values and values["api_key"] is not None:
                target["api_key"] = str(values["api_key"])
    save_config(config)
    return config
