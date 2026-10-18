"""Assemble the exact message list an LLM receives.

The whole point of this module is one guarantee: **the model can always
tell which image came from which socket.**

The mechanism is intentionally boring, because boring is what models
actually follow:

1. The node's image sockets are literally named ``IMAGE1`` … ``IMAGE6``,
   so a task author writes plain prose -- "參考 IMAGE1 的景、IMAGE2 的人、
   IMAGE3 的色調" -- with no placeholder syntax to learn.
2. A manifest turn lists every connected socket, its declared role, and
   its position in the send order, and separately names the sockets the
   task expected but the user did not connect.
3. Each image then travels in its own message whose entire text is the
   tag ``【IMAGE1（景）】``, so no ambiguity exists even for models whose
   chat template renders images before or after the text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .library import IMAGE_SLOTS, Library, Task
from .providers import ChatRequest, ImagePart, Turn

#: Delivery modes exposed on the node.
DELIVERY_PER_IMAGE = "逐張標註"
DELIVERY_SINGLE = "集中單一訊息"
DELIVERY_MODES = (DELIVERY_PER_IMAGE, DELIVERY_SINGLE)

_USER_PROMPT_PATTERN = re.compile(r"\{\{\s*user_prompt\s*\}\}", re.IGNORECASE)


@dataclass
class BuiltPayload:
    """Everything needed to make the call, plus what the UI should report."""

    system: str
    turns: List[Turn]
    notes: List[str] = field(default_factory=list)
    used_slots: List[str] = field(default_factory=list)
    missing_declared: List[str] = field(default_factory=list)

    def to_chat_request(
        self,
        *,
        provider: str,
        base_url: str,
        model: str,
        api_key: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
        think: Optional[bool] = None,
        keep_alive: Optional[str] = None,
        timeout: int = 300,
        extra: Optional[Dict[str, Any]] = None,
    ) -> ChatRequest:
        return ChatRequest(
            provider=provider,
            base_url=base_url,
            model=model,
            system=self.system,
            turns=self.turns,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            think=think,
            keep_alive=keep_alive,
            timeout=timeout,
            extra=extra or {},
        )


def image_tag(slot: str, role: str = "") -> str:
    """The text that travels with one image: ``【IMAGE2（人）】``."""
    role = (role or "").strip()
    return f"【{slot}（{role}）】" if role else f"【{slot}】"


def manifest_text(task: Task, connected: Sequence[str]) -> str:
    """The "which socket is which picture" legend, sent before the images."""
    connected = [slot for slot in connected if slot]
    declared = task.declared_slots()
    missing = [slot for slot in declared if slot not in connected]
    lines: List[str] = ["【本次輸入圖片對照】"]

    if not connected:
        lines.append("本次沒有連接任何圖片，請只根據文字指示完成任務。")
        if missing:
            lines.append(f"（任務原定參考的接口 {_join_slots(missing)} 本次未提供。）")
        return "\n".join(lines)

    lines.append(f"本次送達的圖片共 {len(connected)} 張，依傳送順序對應的接口如下：")
    for index, slot in enumerate(connected, start=1):
        role = task.role_for(slot)
        suffix = f"（{role}）" if role else ""
        lines.append(f"- {slot}{suffix}：第 {index} 張")
    if missing:
        lines.append(
            f"未連接的接口：{_join_slots(missing)} — 本次沒有這些圖片，"
            "請不要臆測或假裝看過它們的內容。"
        )
    else:
        lines.append("本次任務提到的接口都有連上圖片。")
    return "\n".join(lines)


def _join_slots(slots: Sequence[str]) -> str:
    return "、".join(slots)


def task_instruction(task: Task, user_prompt: str, notes: List[str]) -> str:
    """The task's stored prompt, with the optional user-prompt substitution."""
    text = task.prompt or ""
    user_prompt = (user_prompt or "").strip()

    if _USER_PROMPT_PATTERN.search(text):
        if not user_prompt:
            notes.append("TASK 使用了 {{user_prompt}}，但節點的 user_prompt 是空的，該處會留白。")
        return _USER_PROMPT_PATTERN.sub(lambda _match: user_prompt, text)
    return text


def build_payload(
    task: Task,
    library: Library,
    images: Dict[str, ImagePart],
    *,
    user_prompt: str = "",
    system_override: str = "",
    delivery: str = DELIVERY_PER_IMAGE,
) -> BuiltPayload:
    """Build the system prompt and message list for one run.

    ``images`` maps socket name (``IMAGE1`` …) to an already-encoded image.
    """
    notes: List[str] = []
    connected = [slot for slot in IMAGE_SLOTS if slot in images]
    for slot in images:
        if slot not in IMAGE_SLOTS:
            notes.append(f"忽略未知的圖片接口 {slot}。")

    system = (system_override or "").strip() or library.system_for(task)
    manifest = manifest_text(task, connected)
    instruction = task_instruction(task, user_prompt, notes).strip()
    user_prompt = (user_prompt or "").strip()

    # When the author positioned the runtime text themselves, do not also
    # append it -- that would send the same instruction twice.
    placeholder_used = bool(_USER_PROMPT_PATTERN.search(task.prompt or ""))
    append_user = bool(task.append_user_prompt) and bool(user_prompt) and not placeholder_used

    turns: List[Turn] = []
    if delivery == DELIVERY_SINGLE:
        blocks = [manifest]
        if instruction:
            blocks.append(instruction)
        elif not placeholder_used and not append_user:
            notes.append("這個 TASK 沒有提示詞內容，只送出了圖片對照表。")
        if append_user:
            blocks.append(f"【使用者補充指示】\n{user_prompt}")
        turns.append(Turn(text="\n\n".join(blocks), images=[images[slot] for slot in connected]))
    else:
        turns.append(Turn(text=manifest))
        for slot in connected:
            image = images[slot]
            turns.append(Turn(text=image_tag(slot, task.role_for(slot)), images=[image]))
        body = instruction
        if not body and not append_user:
            body = "請依照上述圖片完成任務。"
            notes.append("這個 TASK 沒有提示詞內容，已改用預設指示。")
        if body:
            turns.append(Turn(text=body))
        if append_user:
            turns.append(Turn(text=f"【使用者補充指示】\n{user_prompt}"))

    missing_declared = [slot for slot in task.declared_slots() if slot not in connected]
    if missing_declared:
        notes.append(
            f"TASK 宣告了 {_join_slots(missing_declared)} 的用途，但這些接口沒有連接圖片。"
        )

    return BuiltPayload(
        system=system,
        turns=turns,
        notes=notes,
        used_slots=connected,
        missing_declared=missing_declared,
    )
