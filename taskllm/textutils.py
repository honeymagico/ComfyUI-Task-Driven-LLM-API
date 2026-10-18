"""Small text helpers shared by the payload builder and the node.

Two problems show up constantly when a node's output is fed straight into
an image generator:

* reasoning models leak `` thinking...<｜end▁of▁thinking｜>`` blocks into the answer;
* models wrap the answer in a markdown code fence because it "looks like
  data".

Both are stripped by :func:`clean_output`, which is intentionally
conservative: it only removes a fence when the *entire* answer is one
fenced block, so a prompt that legitimately contains backticks survives.
"""

from __future__ import annotations

import re

#: Reasoning wrappers seen in the wild (Qwen, DeepSeek-R1, Gemma, ...).
_REASONING_PATTERNS = (
    re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<\|thinking\|>.*?<\|/thinking\|>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<\|channel\|>thought.*?<channel\|>", re.DOTALL | re.IGNORECASE),
    re.compile(r"^\s*Thinking\.\.\..*?\.\.\.done thinking\.\s*", re.DOTALL | re.IGNORECASE),
)

#: A fence that starts the answer and closes at the very end.
_WHOLE_FENCE = re.compile(r"^\s*```[^\n`]*\n(.*?)\n?```\s*$", re.DOTALL)

#: Opening pleasantries models like to add before "here is your prompt".
_PREAMBLE_PATTERNS = (
    re.compile(r"^\s*(?:好的|好，|當然|没问题|沒問題)[，,：:].{0,40}?[:：]\s*\n+"),
    re.compile(r"^\s*(?:Sure|Certainly|Of course)[,!]?.{0,60}?[:：]\s*\n+", re.IGNORECASE),
)


def strip_reasoning(text: str) -> str:
    """Remove reasoning blocks from ``text`` (safe when there are none)."""
    if not text:
        return ""
    result = text
    for pattern in _REASONING_PATTERNS:
        result = pattern.sub("", result)
    return result.strip()


def unwrap_code_fence(text: str) -> str:
    """Drop a fence that wraps the whole answer, keeping everything inside."""
    match = _WHOLE_FENCE.match(text or "")
    return match.group(1).strip() if match else (text or "")


def strip_preamble(text: str) -> str:
    """Remove a leading "好的，以下是..." style line.

    Only fires when the pleasantry is followed by a colon and a newline, so
    a legitimate sentence that merely starts with 好的 is left alone.
    """
    result = text or ""
    for pattern in _PREAMBLE_PATTERNS:
        replaced = pattern.sub("", result, count=1)
        if replaced != result:
            return replaced.lstrip()
    return result


def clean_output(text: str, *, remove_preamble: bool = True) -> str:
    """Full cleanup pass applied to every answer before it is returned."""
    result = strip_reasoning(text)
    result = unwrap_code_fence(result)
    if remove_preamble:
        result = strip_preamble(result)
    return result.strip()


def truncate(text: str, limit: int = 400) -> str:
    """Shorten ``text`` for logs and error messages."""
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"
