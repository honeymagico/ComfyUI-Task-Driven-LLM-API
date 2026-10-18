"""One adapter per LLM API.

Adding a provider means writing a small class with two or three methods and
registering it in :data:`PROVIDERS`; the node, the web UI and the routes
all read from that registry, so nothing else needs to change.

Two adapters ship today:

``ollama``
    The native ``/api/chat`` endpoint.  It is the recommended local
    option: images ride along as plain base64 on a message, ``think`` and
    ``keep_alive`` are first-class, and no API key is involved.

``openai_compatible``
    Anything that speaks ``/v1/chat/completions``: LM Studio, llama.cpp
    server, vLLM, OpenRouter, DeepSeek, OpenAI itself.

Both are deliberately synchronous (``requests``).  ComfyUI executes nodes
on a worker thread, never on the aiohttp event loop, so blocking here does
not stall the UI.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from . import config as config_module

# --------------------------------------------------------------------------
# Neutral request / result types shared by every adapter
# --------------------------------------------------------------------------


@dataclass
class ImagePart:
    """One image, already encoded, with the label that disambiguates it."""

    slot: str  #: IMAGE1 … IMAGE6 -- the socket name the user sees
    label: str  #: the text sent immediately next to the image
    data_b64: str
    mime: str
    role: str = ""
    width: int = 0
    height: int = 0
    source_width: int = 0
    source_height: int = 0
    byte_count: int = 0

    def summary(self) -> Dict[str, Any]:
        return {
            "slot": self.slot,
            "role": self.role,
            "label": self.label,
            "mime": self.mime,
            "width": self.width,
            "height": self.height,
            "source_width": self.source_width,
            "source_height": self.source_height,
            "kb": round(self.byte_count / 1024, 1),
            "downscaled": bool(
                self.source_width and (self.width, self.height) != (self.source_width, self.source_height)
            ),
        }


@dataclass
class Turn:
    """One ``user`` message: text plus any images sent with that text."""

    text: str
    images: List[ImagePart] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        return {"text": self.text, "images": [image.summary() for image in self.images]}


@dataclass
class ChatRequest:
    provider: str
    base_url: str
    model: str
    system: str = ""
    turns: List[Turn] = field(default_factory=list)
    api_key: str = ""
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    seed: Optional[int] = None
    think: Optional[bool] = None
    keep_alive: Optional[str] = None
    timeout: int = 300
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatResult:
    text: str
    thinking: str = ""
    model: str = ""
    provider: str = ""
    endpoint: str = ""
    duration_ms: int = 0
    usage: Dict[str, Any] = field(default_factory=dict)
    response: Dict[str, Any] = field(default_factory=dict)


class ProviderError(RuntimeError):
    """A failure worth showing the user verbatim, with a fix attached."""

    def __init__(self, message: str, hint: str = "", *, endpoint: str = "", status: Optional[int] = None):
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.endpoint = endpoint
        self.status = status

    def __str__(self) -> str:  # shown in ComfyUI's error toast
        if self.hint:
            return f"{self.message}\n→ {self.hint}"
        return self.message


# --------------------------------------------------------------------------
# HTTP helpers
# --------------------------------------------------------------------------


def _join(base_url: str, path: str) -> str:
    return (base_url or "").rstrip("/") + "/" + path.lstrip("/")


def chat_endpoint(base_url: str) -> str:
    """Resolve the OpenAI-compatible chat URL from a user-supplied base."""
    base = (base_url or "").rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if re.search(r"/v\d+[a-z]*$", base):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def models_endpoint(base_url: str) -> str:
    base = (base_url or "").rstrip("/")
    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")]
    if re.search(r"/v\d+[a-z]*$", base):
        return base + "/models"
    return base + "/v1/models"


def _requests() -> Any:
    import requests  # noqa: PLC0415

    return requests


def _timeout_pair(timeout: int) -> Tuple[int, int]:
    timeout = max(5, int(timeout or 300))
    return (min(15, timeout), timeout)


def _request(
    method: str,
    url: str,
    *,
    payload: Optional[dict] = None,
    headers: Optional[dict] = None,
    timeout: int = 300,
    provider_label: str = "LLM API",
) -> Tuple[Any, float]:
    """Perform one HTTP call, mapping every failure to a :class:`ProviderError`."""
    requests = _requests()
    started = time.perf_counter()
    try:
        response = requests.request(
            method,
            url,
            json=payload,
            headers=headers,
            timeout=_timeout_pair(timeout),
        )
    except requests.exceptions.ConnectTimeout:
        raise ProviderError(
            f"連線 {provider_label} 逾時：{url}",
            "請確認服務已啟動、位址與連接埠正確，而且沒有被防火牆阻擋。",
            endpoint=url,
        ) from None
    except requests.exceptions.ReadTimeout:
        raise ProviderError(
            f"{provider_label} 在 {timeout} 秒內沒有回覆：{url}",
            "模型可能正在載入或圖片太大；可提高節點的 timeout，或降低 image_max_side。",
            endpoint=url,
        ) from None
    except requests.exceptions.ConnectionError as error:
        detail = str(error)
        if "Name or service not known" in detail or "nodename nor servname" in detail:
            hint = "網域名稱解析失敗，請檢查 base_url 是否拼錯。"
        else:
            hint = "請確認服務已啟動（Ollama：`ollama serve`）、位址與連接埠正確。"
        raise ProviderError(f"無法連線 {provider_label}：{url}", hint, endpoint=url) from None
    except requests.exceptions.RequestException as error:
        raise ProviderError(f"{provider_label} 請求失敗：{error}", endpoint=url) from None

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    body: Any
    try:
        body = response.json()
    except ValueError:
        body = {"_raw": response.text[:2000]}

    if response.status_code >= 400:
        raise ProviderError(
            _error_message(body, response.status_code, provider_label),
            _error_hint(body, response.status_code, provider_label),
            endpoint=url,
            status=response.status_code,
        )
    return body, elapsed_ms


def _error_message(body: Any, status: int, provider_label: str) -> str:
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and error.get("message"):
            return f"{provider_label} 回應錯誤（HTTP {status}）：{error['message']}"
        if isinstance(error, str) and error:
            return f"{provider_label} 回應錯誤（HTTP {status}）：{error}"
        if body.get("message"):
            return f"{provider_label} 回應錯誤（HTTP {status}）：{body['message']}"
        if body.get("_raw"):
            return f"{provider_label} 回應錯誤（HTTP {status}）：{body['_raw'][:300]}"
    return f"{provider_label} 回應錯誤（HTTP {status}）。"


def _error_hint(body: Any, status: int, provider_label: str) -> str:
    text = json.dumps(body, ensure_ascii=False).lower() if isinstance(body, (dict, list)) else str(body).lower()
    if "model" in text and ("not found" in text or "no such" in text or "unknown" in text):
        return "這個模型不存在或尚未下載；請在節點「提示詞庫 → 連線設定」重新整理模型清單（Ollama：`ollama pull <模型>`）。"
    if "unknown field" in text:
        return "伺服器版本不支援某個參數，請把 think 設為「自動」，或更新伺服器版本。"
    if status in (401, 403):
        return "金鑰無效或未設定；請在「連線設定」補上 API Key。"
    if status == 404:
        return "端點不存在；請確認 base_url（Ollama 為 http://127.0.0.1:11434，OpenAI 相容通常以 /v1 結尾）。"
    if status == 429:
        return "被限流了，稍後再試或降低並行數量。"
    if status >= 500:
        return "伺服器端錯誤；請查看服務日誌（Ollama 的 `ollama serve` 視窗）。"
    return ""


# --------------------------------------------------------------------------
# Adapters
# --------------------------------------------------------------------------


class OllamaProvider:
    """Native Ollama ``/api/chat`` -- the recommended local path."""

    id = config_module.PROVIDER_OLLAMA
    label = "Ollama"

    #: Ollama expects bare base64, with the media type implied per message.
    images_as_data_uri = False

    def chat(self, request: ChatRequest) -> ChatResult:
        url = _join(request.base_url, "/api/chat")
        payload: Dict[str, Any] = {
            "model": request.model,
            "messages": self._messages(request),
            "stream": False,
        }
        options: Dict[str, Any] = {}
        if request.temperature is not None:
            options["temperature"] = float(request.temperature)
        if request.max_tokens:
            options["num_predict"] = int(request.max_tokens)
        if request.seed is not None:
            options["seed"] = int(request.seed)
        if options:
            payload["options"] = options
        if request.keep_alive:
            payload["keep_alive"] = request.keep_alive
        if request.think is not None:
            payload["think"] = bool(request.think)

        headers = {"Content-Type": "application/json"}
        if request.api_key:
            # Ollama ignores this, but proxies in front of it may not.
            headers["Authorization"] = f"Bearer {request.api_key}"

        body, elapsed_ms = self._post_with_think_fallback(url, payload, headers, request.timeout)

        message = body.get("message") if isinstance(body, dict) else None
        if not isinstance(message, dict):
            raise ProviderError(
                f"Ollama 回傳了非預期的格式：{str(body)[:300]}",
                "請確認 base_url 指向 Ollama 原生介面（不是 /v1）。",
                endpoint=url,
            )

        usage = {
            key: body.get(key)
            for key in ("prompt_eval_count", "eval_count", "total_duration", "load_duration", "done_reason")
            if body.get(key) is not None
        }
        return ChatResult(
            text=str(message.get("content") or ""),
            thinking=str(message.get("thinking") or ""),
            model=str(body.get("model") or request.model),
            provider=self.id,
            endpoint=url,
            duration_ms=elapsed_ms,
            usage=usage,
            response=body if isinstance(body, dict) else {},
        )

    def _post_with_think_fallback(
        self, url: str, payload: Dict[str, Any], headers: dict, timeout: int
    ) -> Tuple[Any, float]:
        """Retry once without ``think`` when the server predates that field."""
        try:
            return _request("POST", url, payload=payload, headers=headers, timeout=timeout, provider_label=self.label)
        except ProviderError as error:
            if "think" not in payload or "unknown field" not in str(error).lower():
                raise
            retry = {key: value for key, value in payload.items() if key != "think"}
            return _request(
                "POST", url, payload=retry, headers=headers, timeout=timeout, provider_label=self.label
            )

    def _messages(self, request: ChatRequest) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        for turn in request.turns:
            message: Dict[str, Any] = {"role": "user", "content": turn.text}
            if turn.images:
                message["images"] = [image.data_b64 for image in turn.images]
            messages.append(message)
        return messages

    def list_models(self, settings: Dict[str, Any]) -> List[str]:
        base_url = str(settings.get("base_url") or "").rstrip("/")
        body, _elapsed = _request(
            "GET",
            _join(base_url, "/api/tags"),
            timeout=int(settings.get("timeout") or 20),
            provider_label=self.label,
        )
        models = body.get("models") if isinstance(body, dict) else None
        names = []
        if isinstance(models, list):
            for entry in models:
                if isinstance(entry, dict):
                    name = entry.get("name") or entry.get("model")
                    if isinstance(name, str) and name:
                        names.append(name)
        return sorted(dict.fromkeys(names), key=str.lower)

    def probe(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        base_url = str(settings.get("base_url") or "").rstrip("/")
        version_body, elapsed = _request(
            "GET", _join(base_url, "/api/version"), timeout=10, provider_label=self.label
        )
        version = version_body.get("version") if isinstance(version_body, dict) else None
        models = self.list_models(settings)
        return {
            "ok": True,
            "provider": self.id,
            "endpoint": base_url,
            "version": version,
            "models": models,
            "message": f"Ollama {version or ''} 連線正常，找到 {len(models)} 個模型。".strip(),
            "duration_ms": elapsed,
        }


class OpenAICompatProvider:
    """``/v1/chat/completions`` -- cloud APIs and local OpenAI servers."""

    id = config_module.PROVIDER_OPENAI
    label = "OpenAI 相容端點"
    images_as_data_uri = True

    def chat(self, request: ChatRequest) -> ChatResult:
        url = chat_endpoint(request.base_url)
        messages: List[Dict[str, Any]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        for turn in request.turns:
            content: List[Dict[str, Any]] = []
            if turn.text:
                content.append({"type": "text", "text": turn.text})
            for image in turn.images:
                content.append(
                    {"type": "image_url", "image_url": {"url": f"data:{image.mime};base64,{image.data_b64}"}}
                )
            if not content:
                content = [{"type": "text", "text": ""}]
            messages.append({"role": "user", "content": content})

        payload: Dict[str, Any] = {"model": request.model, "messages": messages, "stream": False}
        if request.temperature is not None:
            payload["temperature"] = float(request.temperature)
        if request.max_tokens:
            payload["max_tokens"] = int(request.max_tokens)
        if request.seed is not None:
            payload["seed"] = int(request.seed)
        payload.update(request.extra or {})

        headers = {"Content-Type": "application/json"}
        if request.api_key:
            headers["Authorization"] = f"Bearer {request.api_key}"

        body, elapsed_ms = _request(
            "POST", url, payload=payload, headers=headers, timeout=request.timeout, provider_label=self.label
        )
        choices = body.get("choices") if isinstance(body, dict) else None
        if not isinstance(choices, list) or not choices:
            raise ProviderError(
                f"{self.label} 回傳了沒有 choices 的回應：{str(body)[:300]}",
                "請確認 base_url 與模型名稱是否正確。",
                endpoint=url,
            )

        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, list):  # some gateways return content parts
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        thinking = message.get("reasoning_content") or message.get("reasoning") or ""
        if not content and isinstance(thinking, str) and thinking:
            # Reasoning-only servers (some DeepSeek-R1 builds) leave content
            # empty; prefer an answer, but never return nothing.
            content = choices[0].get("text") or ""

        return ChatResult(
            text=str(content or ""),
            thinking=str(thinking or ""),
            model=str(body.get("model") or request.model),
            provider=self.id,
            endpoint=url,
            duration_ms=elapsed_ms,
            usage=body.get("usage") if isinstance(body.get("usage"), dict) else {},
            response=body if isinstance(body, dict) else {},
        )

    def list_models(self, settings: Dict[str, Any]) -> List[str]:
        base_url = str(settings.get("base_url") or "").rstrip("/")
        headers = {}
        api_key = str(settings.get("api_key") or "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        body, _elapsed = _request(
            "GET",
            models_endpoint(base_url),
            headers=headers,
            timeout=int(settings.get("timeout") or 20),
            provider_label=self.label,
        )
        entries = body.get("data") if isinstance(body, dict) else None
        names = []
        if isinstance(entries, list):
            for entry in entries:
                if isinstance(entry, dict):
                    name = entry.get("id") or entry.get("name")
                    if isinstance(name, str) and name:
                        names.append(name)
                elif isinstance(entry, str):
                    names.append(entry)
        return sorted(dict.fromkeys(names), key=str.lower)

    def probe(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        base_url = str(settings.get("base_url") or "").rstrip("/")
        models = self.list_models(settings)
        return {
            "ok": True,
            "provider": self.id,
            "endpoint": chat_endpoint(base_url),
            "version": None,
            "models": models,
            "message": f"端點連線正常，找到 {len(models)} 個模型。",
            "duration_ms": 0,
        }


PROVIDERS: Dict[str, Any] = {
    OllamaProvider.id: OllamaProvider(),
    OpenAICompatProvider.id: OpenAICompatProvider(),
}


def get_provider(provider_id: str) -> Any:
    if provider_id not in PROVIDERS:
        raise ProviderError(
            f"不認識的 provider：{provider_id}",
            f"目前支援：{', '.join(sorted(PROVIDERS))}。",
        )
    return PROVIDERS[provider_id]


def provider_label(provider_id: str) -> str:
    provider = PROVIDERS.get(provider_id)
    return getattr(provider, "label", provider_id)


# --------------------------------------------------------------------------
# Convenience entry points used by the node and the routes
# --------------------------------------------------------------------------


def chat(request: ChatRequest) -> ChatResult:
    return get_provider(request.provider).chat(request)


def list_models(provider_id: str, settings: Optional[Dict[str, Any]] = None) -> List[str]:
    settings = settings or config_module.provider_settings(provider_id)
    return get_provider(provider_id).list_models(settings)


def probe(provider_id: str, settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    settings = settings or config_module.provider_settings(provider_id)
    provider = get_provider(provider_id)
    if hasattr(provider, "probe"):
        return provider.probe(settings)
    models = provider.list_models(settings)
    return {
        "ok": True,
        "provider": provider_id,
        "endpoint": settings.get("base_url", ""),
        "models": models,
        "message": f"連線正常，找到 {len(models)} 個模型。",
    }


def describe_request(request: ChatRequest) -> Dict[str, Any]:
    """A JSON-safe echo of a request, with image bytes replaced by sizes."""
    return {
        "provider": request.provider,
        "endpoint": (
            _join(request.base_url, "/api/chat")
            if request.provider == config_module.PROVIDER_OLLAMA
            else chat_endpoint(request.base_url)
        ),
        "model": request.model,
        "system": request.system,
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "seed": request.seed,
        "think": request.think,
        "keep_alive": request.keep_alive,
        "messages": [{"role": "user", **turn.summary()} for turn in request.turns],
    }


def render_request_preview(request: ChatRequest) -> str:
    """Human-readable transcript, used by the editor's 預覽 pane."""
    lines: List[str] = []
    if request.system:
        lines.append("═══ system ═══")
        lines.append(request.system)
        lines.append("")
    for index, turn in enumerate(request.turns, start=1):
        header = f"═══ user #{index} ═══"
        lines.append(header)
        if turn.images:
            for image in turn.images:
                lines.append(
                    f"[圖片] {image.slot}"
                    + (f"（{image.role}）" if image.role else "")
                    + f" — {image.width}×{image.height}, {round(image.byte_count / 1024, 1)} KB"
                )
        if turn.text:
            lines.append(turn.text)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def payload_size_hint(request: ChatRequest) -> Dict[str, Any]:
    images = [image for turn in request.turns for image in turn.images]
    return {
        "turns": len(request.turns),
        "images": len(images),
        "image_kb": round(sum(image.byte_count for image in images) / 1024, 1),
        "system_chars": len(request.system or ""),
        "text_chars": sum(len(turn.text or "") for turn in request.turns),
    }
