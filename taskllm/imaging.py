"""Turn a ComfyUI IMAGE tensor into base64 the way vision APIs want it.

ComfyUI hands nodes a ``float32`` torch tensor shaped ``[B, H, W, C]`` with
values in ``0..1``.  Vision models, meanwhile, charge by the pixel: a
1024x1024 JPEG is roughly 1.1k tokens on most providers, and a full-res
phone photo can be an order of magnitude more.  So every image is
downscaled to a configurable longest side (1024 px by default) before it is
encoded, and the caller is told both the sent and the original size so the
node can report what actually happened.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Any, Optional, Tuple

#: Longest side used when the caller does not specify one.
DEFAULT_MAX_SIDE = 1024
DEFAULT_QUALITY = 90


@dataclass
class EncodedImage:
    """One encoded image plus the metadata worth reporting to the user."""

    data_b64: str
    mime: str
    width: int
    height: int
    source_width: int
    source_height: int
    byte_count: int

    def data_uri(self) -> str:
        return f"data:{self.mime};base64,{self.data_b64}"

    def summary(self) -> dict:
        return {
            "mime": self.mime,
            "width": self.width,
            "height": self.height,
            "source_width": self.source_width,
            "source_height": self.source_height,
            "kb": round(self.byte_count / 1024, 1),
            "downscaled": (self.width, self.height) != (self.source_width, self.source_height),
        }


def _numpy() -> Any:
    import numpy  # noqa: PLC0415 - imported lazily so the module stays import-light

    return numpy


def _to_hwc_uint8(image: Any) -> Tuple[Any, int, int]:
    """Normalise torch / numpy / PIL input to an ``H x W x C`` uint8 array."""
    np = _numpy()

    try:  # torch tensors are the ComfyUI norm
        import torch

        if isinstance(image, torch.Tensor):
            array = image.detach().to("cpu").numpy()
        else:
            array = np.asarray(image)
    except Exception:
        array = np.asarray(image)

    if array.dtype == object:
        raise ValueError("不支援的圖片型別（object 陣列）。")

    if array.ndim == 4:  # batch: take the first frame, like every other node
        array = array[0]
    elif array.ndim == 2:  # single-channel grayscale
        array = array[:, :, None]
    if array.ndim != 3:
        raise ValueError(f"圖片張量的維度必須是 2–4 維，收到 {array.ndim} 維。")

    if array.shape[0] in (1, 3, 4) and array.shape[2] not in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))  # CHW -> HWC
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    elif array.shape[2] == 2:
        array = np.repeat(array[:, :, :1], 3, axis=2)
    elif array.shape[2] > 4:
        array = array[:, :, :3]

    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating):
            # ComfyUI floats are 0..1, but tolerate 0..255 floats too.
            peak = float(np.nanmax(array)) if array.size else 1.0
            scale = 255.0 if peak > 1.5 else 1.0
            array = array * scale
        array = np.clip(array, 0, 255).round().astype(np.uint8)

    height, width = int(array.shape[0]), int(array.shape[1])
    if height < 1 or width < 1:
        raise ValueError("圖片尺寸為 0，無法編碼。")
    return np.ascontiguousarray(array), width, height


def encode_image(
    image: Any,
    *,
    fmt: str = "JPEG",
    quality: int = DEFAULT_QUALITY,
    max_side: int = DEFAULT_MAX_SIDE,
) -> EncodedImage:
    """Encode one image for an LLM API request.

    ``max_side`` of 0 or ``None`` keeps the original resolution.
    """
    from PIL import Image  # noqa: PLC0415 - Pillow is always present inside ComfyUI

    array, source_width, source_height = _to_hwc_uint8(image)
    pil = Image.fromarray(array)
    if pil.mode != "RGB":
        pil = pil.convert("RGB")

    target_width, target_height = source_width, source_height
    if max_side and max(int(max_side), 1) > 0 and max(source_width, source_height) > int(max_side):
        limit = int(max_side)
        scale = limit / float(max(source_width, source_height))
        target_width = max(1, int(round(source_width * scale)))
        target_height = max(1, int(round(source_height * scale)))
        pil = pil.resize((target_width, target_height), Image.LANCZOS)

    fmt = (fmt or "JPEG").upper()
    if fmt not in ("JPEG", "PNG", "WEBP"):
        fmt = "JPEG"
    mime = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}[fmt]

    buffer = io.BytesIO()
    save_kwargs: dict = {}
    if fmt == "JPEG":
        save_kwargs = {"quality": int(max(1, min(100, quality))), "optimize": True, "progressive": False}
    elif fmt == "PNG":
        save_kwargs = {"compress_level": 6, "optimize": False}
    elif fmt == "WEBP":
        save_kwargs = {"quality": int(max(1, min(100, quality)))}
    pil.save(buffer, format=fmt, **save_kwargs)
    raw = buffer.getvalue()

    return EncodedImage(
        data_b64=base64.b64encode(raw).decode("ascii"),
        mime=mime,
        width=target_width,
        height=target_height,
        source_width=source_width,
        source_height=source_height,
        byte_count=len(raw),
    )


def estimate_tokens(encoded: Optional[EncodedImage]) -> int:
    """Rough vision-token estimate; only used for the debug readout."""
    if encoded is None:
        return 0
    # Anthropic's rule of thumb: (w * h) / 750.  Close enough for a hint.
    return max(1, int((encoded.width * encoded.height) / 750))
