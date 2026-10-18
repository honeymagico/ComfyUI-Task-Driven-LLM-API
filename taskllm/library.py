"""The TASK prompt library: schema, validation and persistence.

The library is a single JSON document.  A *TASK* is one stored job such as
"compose a prompt from IMAGE1's scene, IMAGE2's person and IMAGE3's colour
grade".  The node shows every enabled task in a dropdown; picking one
supplies the system prompt, the instruction text and a few generation
settings.

Design notes
------------

**No template variables are required.**  The images the user wires into the
node are named ``IMAGE1``…``IMAGE6`` on the sockets themselves, and the
backend labels each image with that same name right before sending it.  A
task author therefore writes plain prose -- "參考 IMAGE1 的景、IMAGE2 的人"
-- and never has to learn a placeholder syntax.  ``{{user_prompt}}`` is
supported as an *optional* convenience for authors who want to control
where the runtime text lands; when it is absent the runtime text is simply
appended as its own final turn.

**Saving never loses work.**  :func:`coerce_library` is deliberately
forgiving: it fills in defaults, drops entries that cannot be a task at
all, and returns human-readable warnings instead of refusing the save.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from . import paths

#: Socket names the node exposes, in the order images are sent.
IMAGE_SLOTS: Tuple[str, ...] = ("IMAGE1", "IMAGE2", "IMAGE3", "IMAGE4", "IMAGE5", "IMAGE6")

#: Placeholder authors may use to position the runtime user prompt.
USER_PROMPT_TOKEN = "{{user_prompt}}"

LIBRARY_VERSION = 1

#: Shipped default system prompt.  It carries the image-mapping contract,
#: which is the one thing the model must be told on every single call --
#: tasks inherit it unless they define their own system prompt.
DEFAULT_SYSTEM = """你是一位專業的視覺分析師與提示詞工程師，負責完成使用者交付的任務。

【圖片輸入約定】
- 節點的圖片接口命名為 IMAGE1、IMAGE2、…，與你收到的圖片一一對應。
- 每張圖片送達時都會附上「【IMAGEn】」標記，標記代表該張圖來自哪個接口。
- 任務內容中若寫到 IMAGE1、IMAGE2 等名稱，指的就是對應接口的那張圖片。
- 對照表中標示「未連接」的接口，本次沒有提供圖片，請不要臆測它的內容。
- 任務若只需要文字，就不會附帶任何圖片。

【輸出規則】
- 直接輸出最終結果，不要加上「好的」、「以下是」這類開場白。
- 除非任務另有要求，否則只輸出結果本身，不要解釋你的推理過程。
- 需要輸出中文時使用繁體中文。"""


def _slugify(text: str, fallback: str = "task") -> str:
    slug = re.sub(r"[^\w\u4e00-\u9fff]+", "-", (text or "").strip().lower(), flags=re.UNICODE)
    slug = re.sub(r"-{2,}", "-", slug).strip("-_")
    return slug or fallback


def _as_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return default


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes", "on", "是", "開啟"):
            return True
        if lowered in ("false", "0", "no", "off", "否", "關閉"):
            return False
    return default


def _as_optional_float(value: Any, low: float, high: float) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return max(low, min(high, number))


def _as_optional_int(value: Any, low: int, high: int) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return None
    return max(low, min(high, number))


def _as_optional_bool(value: Any) -> Optional[bool]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("auto", "自動"):
        return None
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes", "on", "開啟"):
            return True
        if lowered in ("false", "0", "no", "off", "關閉"):
            return False
    return None


@dataclass
class Task:
    """One stored job the user can pick from the node's TASK dropdown."""

    id: str = ""
    title: str = ""
    category: str = ""
    description: str = ""
    system: str = ""
    prompt: str = ""
    append_user_prompt: bool = True
    image_roles: Dict[str, str] = field(default_factory=dict)
    temperature: Optional[float] = None
    num_predict: Optional[int] = None
    think: Optional[bool] = None
    keep_alive: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    enabled: bool = True

    @property
    def label(self) -> str:
        """The exact string shown in the node's TASK dropdown."""
        if self.category:
            return f"{self.category} · {self.title}"
        return self.title

    def role_for(self, slot: str) -> str:
        return (self.image_roles or {}).get(slot, "").strip()

    def declared_slots(self) -> List[str]:
        """Slots the author gave a role to, in canonical order."""
        return [slot for slot in IMAGE_SLOTS if self.role_for(slot)]

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "id": self.id,
            "title": self.title,
            "category": self.category,
            "description": self.description,
            "system": self.system,
            "prompt": self.prompt,
            "append_user_prompt": self.append_user_prompt,
            "enabled": self.enabled,
        }
        if self.image_roles:
            data["image_roles"] = {k: v for k, v in self.image_roles.items() if v}
        for key in ("temperature", "num_predict", "think", "keep_alive", "provider", "model"):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        return data

    @classmethod
    def from_dict(cls, raw: Dict[str, Any], warnings: Optional[List[str]] = None) -> "Task":
        warnings = warnings if warnings is not None else []
        title = _as_str(raw.get("title")).strip()
        prompt = _as_str(raw.get("prompt"))
        if not title and not prompt:
            warnings.append("略過一個沒有標題也沒有內容的 TASK。")
            raise ValueError("empty task")

        task = cls(
            id=_as_str(raw.get("id")).strip(),
            title=title or "未命名 TASK",
            category=_as_str(raw.get("category")).strip(),
            description=_as_str(raw.get("description")).strip(),
            system=_as_str(raw.get("system")),
            prompt=prompt,
            append_user_prompt=_as_bool(raw.get("append_user_prompt"), True),
            temperature=_as_optional_float(raw.get("temperature"), 0.0, 2.0),
            num_predict=_as_optional_int(raw.get("num_predict"), 0, 1_000_000),
            think=_as_optional_bool(raw.get("think")),
            keep_alive=_as_str(raw.get("keep_alive")).strip() or None,
            provider=_as_str(raw.get("provider")).strip() or None,
            model=_as_str(raw.get("model")).strip() or None,
            enabled=_as_bool(raw.get("enabled"), True),
        )

        roles = raw.get("image_roles")
        if isinstance(roles, dict):
            cleaned: Dict[str, str] = {}
            for slot, role in roles.items():
                slot_name = _as_str(slot).strip().upper()
                if slot_name not in IMAGE_SLOTS:
                    if _as_str(role).strip():
                        warnings.append(f"TASK「{task.title}」的圖片角色 {slot_name} 不是 IMAGE1–IMAGE6，已忽略。")
                    continue
                if _as_str(role).strip():
                    cleaned[slot_name] = _as_str(role).strip()
            task.image_roles = cleaned
        elif roles not in (None, ""):
            warnings.append(f"TASK「{task.title}」的 image_roles 必須是物件，已忽略。")

        # An empty system prompt intentionally means "inherit the library's
        # default_system"; only a non-empty string overrides it.
        return task


@dataclass
class Library:
    """The whole prompt library document."""

    version: int = LIBRARY_VERSION
    title: str = "Task-Driven LLM 提示詞庫"
    default_system: str = DEFAULT_SYSTEM
    tasks: List[Task] = field(default_factory=list)

    def enabled_tasks(self) -> List[Task]:
        return [task for task in self.tasks if task.enabled]

    def labels(self) -> List[str]:
        return [task.label for task in self.enabled_tasks()]

    def by_label(self) -> Dict[str, Task]:
        return {task.label: task for task in self.enabled_tasks()}

    def by_id(self) -> Dict[str, Task]:
        return {task.id: task for task in self.tasks}

    def resolve(self, label_or_id: str) -> Optional[Task]:
        if not label_or_id:
            return None
        for task in self.tasks:
            if task.id == label_or_id or task.label == label_or_id:
                return task
        return None

    def system_for(self, task: Task) -> str:
        return task.system.strip() or self.default_system

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "title": self.title,
            "default_system": self.default_system,
            "tasks": [task.to_dict() for task in self.tasks],
        }

    @classmethod
    def from_dict(cls, raw: Any) -> Tuple["Library", List[str]]:
        return coerce_library(raw)


def coerce_library(raw: Any) -> Tuple[Library, List[str]]:
    """Build a :class:`Library` from untrusted JSON plus a list of warnings."""
    warnings: List[str] = []

    if isinstance(raw, list):
        raw = {"tasks": raw}
    if not isinstance(raw, dict):
        raise ValueError("提示詞庫的最外層必須是 JSON 物件（{ ... }）或 TASK 陣列（[ ... ]）。")

    raw_tasks = raw.get("tasks")
    if raw_tasks is None:
        raw_tasks = []
    if not isinstance(raw_tasks, list):
        raise ValueError("提示詞庫的 tasks 欄位必須是陣列。")

    library = Library(
        version=_as_optional_int(raw.get("version"), 1, 999) or LIBRARY_VERSION,
        title=_as_str(raw.get("title")).strip() or "Task-Driven LLM 提示詞庫",
        default_system=_as_str(raw.get("default_system")) if raw.get("default_system") is not None else DEFAULT_SYSTEM,
    )
    if not library.default_system.strip():
        library.default_system = DEFAULT_SYSTEM

    tasks: List[Task] = []
    seen_ids: Dict[str, int] = {}
    seen_labels: Dict[str, int] = {}

    for index, entry in enumerate(raw_tasks):
        if not isinstance(entry, dict):
            warnings.append(f"略過第 {index + 1} 個項目：它不是物件。")
            continue
        try:
            task = Task.from_dict(entry, warnings)
        except ValueError:
            continue

        base_id = task.id or _slugify(task.title or task.prompt, fallback=f"task-{index + 1}")
        count = seen_ids.get(base_id, 0)
        seen_ids[base_id] = count + 1
        task.id = base_id if count == 0 else f"{base_id}-{count + 1}"

        base_label = task.label
        label_count = seen_labels.get(base_label, 0)
        if label_count:
            # Dropdown entries must be unique or the user cannot tell them
            # apart; disambiguate rather than silently shadowing one.
            task.title = f"{task.title} ({label_count + 1})"
            warnings.append(f"TASK 標題「{base_label}」重複，已改名為「{task.label}」。")
        seen_labels[task.label] = label_count + 1

        if not task.prompt.strip() and not task.system.strip():
            warnings.append(f"TASK「{task.title}」沒有提示詞內容。")
        tasks.append(task)

    library.tasks = tasks
    return library, warnings


def load_library(path: Optional[Path] = None) -> Tuple[Library, List[str]]:
    """Read the on-disk library, seeding it from the shipped default if needed."""
    target = Path(path) if path is not None else paths.tasks_path()
    if not target.exists():
        seed = paths.default_tasks_path()
        if seed.exists() and seed != target:
            library, warnings = _read(seed)
            try:
                save_library(library, target)
            except Exception:
                pass
            return library, warnings
        return Library(), []
    return _read(target)


def _read(path: Path) -> Tuple[Library, List[str]]:
    try:
        raw = paths.read_json(path)
    except FileNotFoundError:
        return Library(), [f"找不到提示詞庫檔案：{path}"]
    except Exception as error:  # pragma: no cover - corrupt file path
        return Library(), [f"讀取提示詞庫失敗（{path.name}）：{error}"]
    return coerce_library(raw)


def save_library(library: Library, path: Optional[Path] = None) -> Path:
    """Write the library, keeping a rolling ``.bak`` of the previous version."""
    target = Path(path) if path is not None else paths.tasks_path()
    if target.exists():
        try:
            backup = target.with_suffix(target.suffix + ".bak")
            backup.write_bytes(target.read_bytes())
        except Exception:
            pass
    paths.atomic_write_json(target, library.to_dict())
    return target


def task_labels(path: Optional[Path] = None) -> List[str]:
    """Dropdown entries for the node.

    Never raises: a broken library must not stop ComfyUI from starting, so
    the failure is reported as a single, self-explanatory entry instead.
    """
    try:
        library, warnings = load_library(path)
    except Exception as error:  # pragma: no cover - defensive
        return [f"（提示詞庫讀取失敗：{error}）"]

    labels = library.labels()
    if labels:
        return labels

    for warning in warnings:
        if warning.startswith(("讀取提示詞庫失敗", "找不到提示詞庫")):
            return [f"（{warning}）"]
    return ["（提示詞庫是空的，請按節點上的「提示詞庫」按鈕新增 TASK）"]


def is_placeholder_label(label: str) -> bool:
    return isinstance(label, str) and label.startswith("（") and label.endswith("）")


def iter_slots() -> Iterable[str]:
    return IMAGE_SLOTS
