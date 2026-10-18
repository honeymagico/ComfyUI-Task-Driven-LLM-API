"""Where the node's user data lives.

Two kinds of file are persisted:

* the prompt library (``prompts/tasks.json``) -- the TASKs the user picks
  from, plus the pristine seed ``prompts/tasks.default.json``;
* ``config.json`` -- provider base URLs, API keys and defaults.

Both are *user data*, but they sit next to the code so that "open the
library folder" is a single, obvious place.  When the custom-nodes
directory is not writable -- a common deployment is ComfyUI running as a
service account on a read-only checkout -- every write transparently
falls back to ComfyUI's per-user directory instead of failing.

``locate()`` reports which of the two is actually in use so the web UI
can tell the user where their library really is.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

#: Repository / node root: the folder that contains ``__init__.py``.
NODE_DIR = Path(__file__).resolve().parent.parent

#: Folder name used under ComfyUI's user directory when we have to fall back.
FALLBACK_NAMESPACE = "task_driven_llm"

_resolved: Dict[str, Path] = {}


def _user_dir() -> Optional[Path]:
    """ComfyUI's per-user directory, or ``None`` outside ComfyUI.

    ``get_user_directory`` only exists on newer ComfyUI builds; older ones
    expose ``get_user_dir``.  We check both and never let a failure escape,
    because this runs during custom-node import.
    """
    try:
        import folder_paths  # type: ignore
    except Exception:
        return None
    for attribute in ("get_user_directory", "get_user_dir"):
        getter = getattr(folder_paths, attribute, None)
        if not callable(getter):
            continue
        try:
            return Path(getter())
        except Exception:
            return None
    return None


def _directory_is_writable(path: Path) -> bool:
    """True when ``path`` exists (or can be created) and accepts a new file."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        handle, probe = tempfile.mkstemp(prefix=".write-probe-", dir=str(path))
        os.close(handle)
        os.unlink(probe)
        return True
    except Exception:
        return False


def writable_root() -> Path:
    """The base directory user data is written to.

    ``NODE_DIR`` whenever the custom-nodes checkout is writable, otherwise
    ``<comfy user dir>/task_driven_llm``, and as a last resort a temp
    directory -- so the node still runs (with a clear report from
    :func:`locate`) on exotic setups.
    """
    if "root" in _resolved:
        return _resolved["root"]

    if _directory_is_writable(NODE_DIR):
        _resolved["root"] = NODE_DIR
        return NODE_DIR

    user_dir = _user_dir()
    if user_dir is not None:
        fallback = user_dir / FALLBACK_NAMESPACE
        if _directory_is_writable(fallback):
            _resolved["root"] = fallback
            return fallback

    temporary = Path(tempfile.gettempdir()) / FALLBACK_NAMESPACE
    temporary.mkdir(parents=True, exist_ok=True)
    _resolved["root"] = temporary
    return temporary


def writable_dir(name: str) -> Path:
    """Return ``writable_root()/name``, creating it."""
    key = f"dir:{name}"
    if key not in _resolved:
        path = writable_root() / name
        path.mkdir(parents=True, exist_ok=True)
        _resolved[key] = path
    return _resolved[key]


def prompts_dir() -> Path:
    return writable_dir("prompts")


def backups_dir() -> Path:
    path = prompts_dir() / "backups"
    path.mkdir(parents=True, exist_ok=True)
    return path


def tasks_path() -> Path:
    """The live prompt library, seeded from ``tasks.default.json`` if absent."""
    path = prompts_dir() / "tasks.json"
    if not path.exists():
        seed = default_tasks_path()
        if seed.exists():
            try:
                shutil.copyfile(seed, path)
            except Exception:
                pass
    return path


def default_tasks_path() -> Path:
    """The pristine shipped library, also used by "restore built-in examples"."""
    return prompts_dir() / "tasks.default.json"


def config_path() -> Path:
    return writable_root() / "config.json"


def locate() -> Dict[str, Any]:
    """Describe the on-disk layout for the web UI ("open folder" button)."""
    root = writable_root()
    return {
        "node_dir": str(NODE_DIR),
        "data_root": str(root),
        "prompts_dir": str(prompts_dir()),
        "tasks_path": str(tasks_path()),
        "default_tasks_path": str(default_tasks_path()),
        "config_path": str(config_path()),
        "node_dir_writable": _directory_is_writable(NODE_DIR),
        "prompts_dir_writable": _directory_is_writable(prompts_dir()),
        "using_fallback": root != NODE_DIR,
    }


def atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON via a temp file + rename so a crash can never truncate it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    handle, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)
