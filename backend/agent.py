"""
LangGraph + LangChain agent: tools (MongoDB vector + embeddings), filesystem plans, CLI.

Environment:

- ``GEMINI_API_KEY`` — required.
- ``MONGO_URI`` — for vector tools.
- ``YHACKS_FS_ROOT`` — optional; all FS tools confine paths under this directory (default: cwd).

Plan JSON (``preview_plan`` / ``execute_plan``): a JSON array of objects:

- ``{"action": "create_folder", "path": "relative/path"}``
- ``{"action": "move_file", "from": "a.txt", "to": "dir/a.txt"}`` — optional ``mongo_id`` (24-char
  hex ``_id``) updates that document only; otherwise MongoDB matches on ``filepath`` after the move.
- ``{"action": "remove_file", "path": "old.txt"}`` — moves the file to the **system Trash** (needs
  ``pip install send2trash``) and removes matching MongoDB rows. Provide ``path`` and/or ``mongo_id``;
  if only ``mongo_id``, the filesystem path is read from the indexed document. On **macOS** and **Linux**,
  ``undo_last_action`` can restore the file from Trash and re-index; on other platforms undo may be unavailable.
- ``{"action": "add_file", "path": "doc.pdf", "description": "optional"}`` — indexes an existing file
  with ``ingest_file_to_db`` (embedding + insert). Undo removes the new MongoDB row only (file stays).
- ``{"action": "remove_folder", "path": "relative/dir"}`` — moves the folder to **Trash** (same as
  ``remove_file``) and removes MongoDB rows under that path. **Not undoable** via the agent stack.

    pip install langchain-core langchain-google-genai langgraph
    python agent.py

**Plans:** ``execute_plan`` always runs the same preview as ``preview_plan`` before executing.
Use ``dry_run=True`` on ``execute_plan`` for preview only. Other tools (e.g. ``semantic_file_search``,
``ask_about_files`` for direct Q&A on local file contents via Gemini) do not auto-preview.

Add tools: define ``@tool``, append to ``AGENT_TOOLS``, document parameters in the docstring.
"""

from __future__ import annotations

import json
import mimetypes
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

try:
    import magic
except ImportError:  # pragma: no cover
    magic = None  # type: ignore[misc, assignment]

# Ensure ``backend/`` is on path when running ``python agent.py`` from any cwd.
_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from bson.objectid import ObjectId
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

# --- Backend modules (sibling imports; cwd should be backend/) ---
from add_element import ingest_file_to_db
from query_elements import similarity_search_with_score
from update_element import update_entries, update_filepath_by_id

import remove_elements as remove_mod

try:
    from send2trash import send2trash
except ImportError as _e:  # pragma: no cover
    send2trash = None  # type: ignore[misc, assignment]

DEFAULT_AGENT_SYSTEM = (
    "You help with file organization and MongoDB vector indexing. "
    "For JSON plans, call execute_plan: it shows a plan preview before running steps. "
    "Use execute_plan(plan_json, dry_run=True) to preview without side effects. "
    "preview_plan is optional; it duplicates what execute_plan shows first. "
    "Use ask_about_files to answer questions about one or more local files (paths under the project root). "
    "Removals use the system Trash (send2trash). On macOS/Linux, undo_last_action can restore "
    "remove_file steps when the tool records an undo batch; do not claim undo worked unless it did."
)


# ---------------------------------------------------------------------------
# Filesystem root + undo stack (used by create_folder / plan / undo)
# ---------------------------------------------------------------------------

# All FS tools confine paths under this directory (set YHACKS_FS_ROOT to your project root).
_undo_stack: list[list[dict[str, Any]]] = []

# ``ask_about_files``: Gemini / LangChain practical limits per request
_ASK_MAX_FILES = 10
_ASK_MAX_BYTES_PER_FILE = 100 * 1024 * 1024


def _fs_root() -> Path:
    raw = os.environ.get("YHACKS_FS_ROOT", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return Path.cwd().resolve()


def _resolve_under_root(user_path: str) -> Path:
    """Resolve path; must stay under ``_fs_root()``."""
    root = _fs_root()
    p = Path(user_path).expanduser()
    full = (root / p).resolve() if not p.is_absolute() else p.resolve()
    try:
        full.relative_to(root)
    except ValueError as e:
        raise ValueError(
            f"Path must be under project root {root}. Got: {full}"
        ) from e
    return full


def _mime_for_file_path(path: Path) -> str:
    if magic is not None:
        try:
            return magic.from_file(str(path), mime=True)
        except Exception:
            pass
    mt, _ = mimetypes.guess_type(path.name)
    return mt or "application/octet-stream"


def _parse_file_paths_arg(raw: str) -> list[str]:
    s = raw.strip()
    if not s:
        raise ValueError("file_paths is empty")
    if s.startswith("["):
        data = json.loads(s)
        if not isinstance(data, list):
            raise ValueError("file_paths JSON must be an array of strings")
        out = [str(x).strip() for x in data if str(x).strip()]
        if not out:
            raise ValueError("file_paths array has no paths")
        return out
    return [s]


def load_google_api_key() -> str:
    raw = os.environ.get("GEMINI_API_KEY")
    key = raw.strip() if raw else ""
    if not key:
        raise ValueError(
            "Set GEMINI_API_KEY in the environment before running the agent."
        )
    if raw and ("\n" in raw or "\r" in raw):
        raise ValueError("API key must be a single line with no newlines.")
    return key


def _normalize_mongo_id(raw: str | None) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip()
    return s or None


def _ensure_send2trash() -> None:
    if send2trash is None:
        raise RuntimeError(
            "send2trash is not installed. Run: pip install send2trash "
            "(see backend/requirements-agent.txt)"
        )


def _darwin_trash_dir_for_path(source: Path) -> Path:
    """User Trash directory for a path on macOS (boot volume vs external volumes)."""
    source = source.resolve()
    home = Path.home().resolve()
    try:
        source.relative_to(home)
        return home / ".Trash"
    except ValueError:
        pass
    try:
        if os.stat(source).st_dev == os.stat(home).st_dev:
            return home / ".Trash"
    except OSError:
        pass
    p = source
    while p.parent != p:
        p = p.parent
    return p / ".Trashes" / str(os.getuid())


def _trash_watch_dirs_for_path(source: Path) -> list[Path]:
    """Directories to snapshot before/after send2trash so we can locate the trashed item."""
    if sys.platform == "darwin":
        primary = _darwin_trash_dir_for_path(source)
        home_trash = Path.home() / ".Trash"
        if primary.resolve() == home_trash.resolve():
            return [primary]
        return [primary, home_trash]
    if sys.platform.startswith("linux"):
        xdg_data = os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))
        return [Path(xdg_data) / "Trash" / "files"]
    return []


def _iter_trash_entries(d: Path) -> frozenset[Path]:
    if not d.is_dir():
        return frozenset()
    try:
        return frozenset(d.iterdir())
    except OSError:
        return frozenset()


def _send2trash_and_get_trash_item_path(abs_path: str) -> str | None:
    """Send path to system Trash via send2trash; return absolute path of the item inside Trash, or None."""
    _ensure_send2trash()
    src = Path(abs_path).resolve()
    if not src.exists():
        raise FileNotFoundError(abs_path)
    watch_dirs = _trash_watch_dirs_for_path(src)
    if not watch_dirs:
        send2trash(str(src))  # type: ignore[misc]
        return None
    before: dict[Path, frozenset[Path]] = {d: _iter_trash_entries(d) for d in watch_dirs}
    send2trash(str(src))  # type: ignore[misc]
    basename = src.name
    for d in watch_dirs:
        new_items = _iter_trash_entries(d) - before.get(d, frozenset())
        if not new_items:
            continue
        if len(new_items) == 1:
            return str(next(iter(new_items)).resolve())
        exact = [p for p in new_items if p.name == basename]
        if len(exact) == 1:
            return str(exact[0].resolve())
        return str(max(new_items, key=lambda p: p.stat().st_mtime_ns).resolve())
    return None


def _filepath_from_mongo_id(mid: str) -> str:
    """Return stored filepath for an indexed document, or raise."""
    try:
        oid = ObjectId(mid)
    except Exception as e:
        raise ValueError(f"Invalid mongo_id: {mid!r}") from e
    doc = remove_mod.collection.find_one({"_id": oid}, {"filepath": 1})
    if not doc or not doc.get("filepath"):
        raise ValueError(
            f"No indexed document with _id={mid!r}, or it has no filepath field."
        )
    return str(doc["filepath"])


def _mongo_sync_after_move(
    old_abs: str,
    new_abs: str,
    mongo_id: str | None = None,
) -> int:
    """Point MongoDB at ``new_abs`` after a file move.

    If ``mongo_id`` is set, updates that single document by ``_id`` (preferred).
    Otherwise updates every document whose ``filepath`` equals ``old_abs``.
    Returns modified_count.
    """
    new_abs = os.path.abspath(new_abs)
    mid = _normalize_mongo_id(mongo_id)
    if mid:
        try:
            ObjectId(mid)
        except Exception as e:
            raise ValueError(f"Invalid mongo_id: {mongo_id!r}") from e
        _m, modified = update_filepath_by_id(mid, new_abs)
        return modified
    old_abs = os.path.abspath(old_abs)
    set_fields: dict[str, Any] = {
        "filepath": new_abs,
        "filename": os.path.basename(new_abs),
    }
    _m, modified = update_entries({"filepath": old_abs}, set_fields, many=True)
    return modified


def _parse_plan_json(plan_json: str) -> list[dict[str, Any]]:
    data = json.loads(plan_json)
    if not isinstance(data, list):
        raise ValueError("Plan must be a JSON array of steps.")
    return data


def _describe_step(step: dict[str, Any], i: int) -> str:
    action = (step.get("action") or "").strip()
    if action == "create_folder":
        return f"{i}. create_folder: {step.get('path', '?')}"
    if action == "move_file":
        mid = step.get("mongo_id") or step.get("mongo_object_id")
        extra = f" mongo_id={mid!r}" if mid else ""
        return f"{i}. move_file: {step.get('from', '?')} -> {step.get('to', '?')}{extra}"
    if action == "remove_file":
        mid = step.get("mongo_id") or step.get("mongo_object_id")
        p = step.get("path")
        extra = f" mongo_id={mid!r}" if mid else ""
        path_part = p if (p and str(p).strip()) else "(path from DB if mongo_id)"
        return f"{i}. remove_file: {path_part}{extra}"
    if action == "add_file":
        return f"{i}. add_file: {step.get('path', '?')} desc={step.get('description', '')!r}"
    if action == "remove_folder":
        return f"{i}. remove_folder: {step.get('path', '?')}"
    return f"{i}. unknown action: {step!r}"


def _apply_create_folder(path_str: str) -> list[dict[str, Any]]:
    """Create directory; return undo frames (newest undo last in list for this op)."""
    path = _resolve_under_root(path_str)
    path.mkdir(parents=True, exist_ok=True)
    root = _fs_root()
    return [{"op": "remove_empty_dirs_upward", "leaf": str(path), "root": str(root)}]


def _apply_move_file(
    from_str: str,
    to_str: str,
    mongo_id: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    src = _resolve_under_root(from_str)
    dst = _resolve_under_root(to_str)
    if not src.exists():
        raise FileNotFoundError(f"Source does not exist: {src}")
    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    old_abs = str(src.resolve())
    shutil.move(old_abs, str(dst.resolve()))
    new_abs = str(dst.resolve())
    n = _mongo_sync_after_move(old_abs, new_abs, mongo_id=mongo_id)
    mid = _normalize_mongo_id(mongo_id)
    undo: dict[str, Any] = {"op": "move_file", "from": new_abs, "to": old_abs}
    if mid:
        undo["mongo_id"] = mid
    how = "by _id" if mid else "by filepath"
    msg = f"Moved: {old_abs} -> {new_abs} (MongoDB rows updated: {n}, {how})"
    return msg, [undo]


def _apply_remove_file(
    path_str: str | None,
    mongo_id: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Move file to system Trash (send2trash) and remove MongoDB rows; undo restores from Trash + re-index."""
    mid = _normalize_mongo_id(mongo_id)
    if not path_str and mid:
        path_str = _filepath_from_mongo_id(mid)
    if not path_str:
        raise ValueError(
            "remove_file needs a filesystem path in the plan, or a mongo_id to look up filepath "
            "from the index."
        )
    p = _resolve_under_root(path_str)
    if not p.is_file():
        raise FileNotFoundError(f"Not a file or missing: {p}")
    if mid:
        try:
            ObjectId(mid)
        except Exception as e:
            raise ValueError(f"Invalid mongo_id: {mongo_id!r}") from e
    abs_path = str(p.resolve())
    trash_item = _send2trash_and_get_trash_item_path(abs_path)
    deleted = 0
    if mid:
        oid = ObjectId(mid)
        r = remove_mod.collection.delete_one({"_id": oid})
        deleted = r.deleted_count
    else:
        r = remove_mod.collection.delete_many({"filepath": abs_path})
        deleted = r.deleted_count
    undo: list[dict[str, Any]] = []
    if trash_item:
        undo.append(
            {
                "op": "restore_from_system_trash",
                "trash_path": trash_item,
                "original_path": abs_path,
            }
        )
        extra = f" Trashed copy: {trash_item}. undo_last_action can restore."
    else:
        extra = (
            " Could not resolve the trashed item path for undo (common on Windows); "
            "restore manually from Trash if needed."
        )
    msg = (
        f"Moved file to system Trash: {abs_path}; MongoDB document(s) removed: {deleted}.{extra}"
    )
    return msg, undo


def _apply_add_file(path_str: str, description: str = "") -> tuple[str, list[dict[str, Any]]]:
    """Index existing file via ingest_file_to_db. Undo = remove inserted Mongo row only."""
    p = _resolve_under_root(path_str)
    if not p.is_file():
        raise FileNotFoundError(f"Not a file or missing: {p}")
    oid = ingest_file_to_db(str(p.resolve()), description or None)
    if oid is None:
        raise RuntimeError("Ingest failed (embedding or insert). Check logs.")
    oid_s = str(oid)
    msg = f"Indexed file {p.resolve()} (_id={oid_s})"
    undo: list[dict[str, Any]] = [{"op": "remove_mongo_by_id", "mongo_id": oid_s}]
    return msg, undo


def _mongo_delete_files_under_folder(folder_abs: str) -> int:
    """Delete vector rows for any indexed file whose filepath is inside ``folder_abs`` (recursive)."""
    folder_abs = os.path.abspath(folder_abs).rstrip(os.sep)
    root = str(_fs_root().resolve())
    if folder_abs == root:
        raise ValueError("Refusing to delete MongoDB entries for the entire project root.")
    prefix = folder_abs + os.sep
    pattern = "^" + re.escape(prefix)
    r = remove_mod.collection.delete_many({"filepath": {"$regex": pattern}})
    return r.deleted_count


def _apply_remove_folder(path_str: str) -> tuple[str, list[dict[str, Any]]]:
    """Move directory to Trash and remove MongoDB rows for files under it."""
    _ensure_send2trash()
    p = _resolve_under_root(path_str)
    root = _fs_root().resolve()
    rp = p.resolve()
    if not rp.is_dir():
        raise NotADirectoryError(f"Not a directory or missing: {rp}")
    if rp == root:
        raise ValueError("Refusing to remove the project root directory.")
    n_mongo = _mongo_delete_files_under_folder(str(rp))
    send2trash(str(rp))
    msg = (
        f"Moved folder to Trash: {rp}; MongoDB document(s) deleted: {n_mongo}. "
        "Restore from Finder Trash if needed."
    )
    return msg, []


def _execute_one_step(step: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    action = (step.get("action") or "").strip()
    if action == "create_folder":
        p = step.get("path")
        if not p:
            raise ValueError("create_folder requires 'path'.")
        undos = _apply_create_folder(str(p))
        return f"Created folder: {p}", undos
    if action == "move_file":
        f = step.get("from")
        t = step.get("to")
        if not f or not t:
            raise ValueError("move_file requires 'from' and 'to'.")
        mid = step.get("mongo_id") or step.get("mongo_object_id")
        return _apply_move_file(str(f), str(t), mongo_id=str(mid) if mid else None)
    if action == "remove_file":
        pth = step.get("path")
        mid = step.get("mongo_id") or step.get("mongo_object_id")
        if not pth and not mid:
            raise ValueError(
                "remove_file requires 'path' and/or 'mongo_id' (path can be omitted if mongo_id "
                "is set — filepath is read from MongoDB)."
            )
        return _apply_remove_file(
            str(pth) if pth else None,
            mongo_id=str(mid) if mid else None,
        )
    if action == "add_file":
        pth = step.get("path")
        if not pth:
            raise ValueError("add_file requires 'path'.")
        desc = step.get("description") or ""
        return _apply_add_file(str(pth), description=str(desc))
    if action == "remove_folder":
        pth = step.get("path")
        if not pth:
            raise ValueError("remove_folder requires 'path'.")
        return _apply_remove_folder(str(pth))
    raise ValueError(
        f"Unknown action: {action!r}. "
        "Use create_folder, move_file, remove_file, add_file, or remove_folder."
    )


def _run_undo_step(u: dict[str, Any]) -> None:
    op = u.get("op")
    if op == "remove_empty_dirs_upward":
        leaf = Path(u["leaf"])
        root = Path(u["root"])
        p = leaf
        while p != root and p.exists() and p.is_dir():
            try:
                p.rmdir()
            except OSError:
                break
            p = p.parent
    elif op == "move_file":
        src = _resolve_under_root(u["from"])
        dst = _resolve_under_root(u["to"])
        if not src.exists():
            raise FileNotFoundError(f"Undo move: missing {src}")
        if dst.exists():
            raise FileExistsError(f"Undo move: destination exists {dst}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        old_abs = str(src.resolve())
        shutil.move(old_abs, str(dst.resolve()))
        new_abs = str(dst.resolve())
        mid = u.get("mongo_id")
        _mongo_sync_after_move(old_abs, new_abs, mongo_id=mid if mid else None)
    elif op == "remove_mongo_by_id":
        mid = u.get("mongo_id")
        if not mid:
            raise ValueError("remove_mongo_by_id missing mongo_id")
        r = remove_mod.collection.delete_one({"_id": ObjectId(mid)})
        if not r.deleted_count:
            raise RuntimeError(f"No MongoDB document with _id={mid!r} (already undone?)")
    elif op == "restore_from_system_trash":
        trash_path = Path(u["trash_path"])
        original = Path(u["original_path"])
        if not trash_path.exists():
            raise FileNotFoundError(f"Undo restore: item not in Trash: {trash_path}")
        original.parent.mkdir(parents=True, exist_ok=True)
        if original.exists():
            raise FileExistsError(f"Undo restore: destination already exists {original}")
        shutil.move(str(trash_path.resolve()), str(original.resolve()))
        oid = ingest_file_to_db(str(original.resolve()), None)
        if oid is None:
            raise RuntimeError(
                "Restored file from Trash but re-indexing failed; check logs or re-add the file."
            )
    else:
        raise ValueError(f"Unknown undo op: {op!r}")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
def semantic_file_search(query: str, k: int = 5) -> str:
    """Search indexed files by natural-language meaning (vector similarity on Gemini embeddings).

    Returns filenames, paths, MIME types, and scores. Requires Atlas vector index READY and
    documents in ``yhacks.files``.
    """
    try:
        pairs = similarity_search_with_score(query=query, k=max(1, min(int(k), 20)))
    except Exception as e:
        return f"Search failed: {e}"
    if not pairs:
        return (
            "No matches (empty index, connection issue, or no embedded documents). "
            "Check MONGO_URI and that the vector index exists."
        )
    lines = []
    for i, (doc, score) in enumerate(pairs, 1):
        oid = doc.get("_id")
        oid_s = str(oid) if oid is not None else "?"
        lines.append(
            f"{i}. score={score:.4f} | _id={oid_s} | {doc.get('filename', '?')} | {doc.get('file_type', '?')}\n"
            f"   path: {doc.get('filepath', '?')}"
        )
    return "\n".join(lines)


@tool
def ask_about_files(question: str, file_paths: str) -> str:
    """Read one or more files from disk and answer ``question`` using Gemini (multimodal ``ChatGoogleGenerativeAI``).

    Paths must be under the project root (``YHACKS_FS_ROOT`` or cwd). ``file_paths`` is either:
    - A JSON array of relative paths, e.g. ``["notes/a.pdf", "img/b.png"]``
    - A single relative path as plain text, e.g. ``"docs/readme.txt"``

    Up to 10 files per call; each file max 100 MiB. You can attach multiple files in one request.
    """
    try:
        paths = _parse_file_paths_arg(file_paths)
    except Exception as e:
        return f"Invalid file_paths: {e}"
    if len(paths) > _ASK_MAX_FILES:
        return f"At most {_ASK_MAX_FILES} files per call (got {len(paths)})."
    q = question.strip()
    if not q:
        return "question is empty."
    try:
        api_key = load_google_api_key()
    except ValueError as e:
        return str(e)
    model_name = os.environ.get("AGENT_MODEL", "gemini-2.5-flash")
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0,
        google_api_key=api_key,
    )
    content: list[dict[str, Any]] = []
    labels: list[str] = []
    try:
        for rel in paths:
            p = _resolve_under_root(rel)
            if not p.is_file():
                raise FileNotFoundError(f"Not a file or missing: {p}")
            data = p.read_bytes()
            if len(data) > _ASK_MAX_BYTES_PER_FILE:
                raise ValueError(
                    f"File too large (max {_ASK_MAX_BYTES_PER_FILE // (1024 * 1024)} MiB): {p}"
                )
            mime = _mime_for_file_path(p)
            content.append({"type": "media", "mime_type": mime, "data": data})
            labels.append(f"{rel} ({mime})")
    except Exception as e:
        return f"Failed to read files: {e}"

    preamble = (
        f"The user attached {len(content)} file(s): {', '.join(labels)}.\n"
        "Answer using the file contents when relevant. If they are not enough, say so.\n\n"
        f"Question:\n{q}"
    )
    parts: list[dict[str, Any]] = [{"type": "text", "text": preamble}]
    parts.extend(content)
    try:
        resp = llm.invoke([HumanMessage(content=parts)])
    except Exception as e:
        return f"Gemini request failed: {e}"
    t = getattr(resp, "text", None)
    if isinstance(t, str) and t.strip():
        return t
    out = resp.content
    if isinstance(out, str):
        return out
    if isinstance(out, list):
        chunks: list[str] = []
        for block in out:
            if isinstance(block, dict) and block.get("type") == "text":
                chunks.append(str(block.get("text", "")))
            elif isinstance(block, str):
                chunks.append(block)
        return "".join(chunks) if chunks else str(out)
    return str(out)


@tool
def trash_file(path: str = "", mongo_id: str = "") -> str:
    """Move a file to the **system Trash** (send2trash) and remove its vector row.

    Provide ``path`` under the project root and/or ``mongo_id``. If only ``mongo_id`` is set, the path is read from MongoDB.
    On macOS/Linux, ``undo_last_action`` may restore the file and re-index when the Trash path was recorded.
    """
    p = str(path).strip()
    m = str(mongo_id).strip()
    try:
        msg, frags = _apply_remove_file(
            p if p else None,
            mongo_id=m if m else None,
        )
    except Exception as e:
        return f"Failed: {e}"
    _undo_stack.append(frags)
    return (
        msg
        + f" Undo batch recorded ({len(frags)} atomic step(s)). Stack depth: {len(_undo_stack)}."
    )


def _format_plan_preview(steps: list[dict[str, Any]]) -> str:
    """Human-readable preview for a parsed plan (no side effects)."""
    lines = [
        f"Project root: {_fs_root()}",
        f"Steps: {len(steps)}",
        "",
    ]
    for i, step in enumerate(steps, 1):
        lines.append(_describe_step(step, i))
        try:
            act = (step.get("action") or "").strip()
            if act == "create_folder":
                p = _resolve_under_root(str(step.get("path", "")))
                lines.append(f"   -> mkdir: {p}")
            elif act == "move_file":
                a = _resolve_under_root(str(step.get("from", "")))
                b = _resolve_under_root(str(step.get("to", "")))
                exists = a.exists()
                mid = step.get("mongo_id") or step.get("mongo_object_id")
                mongo_note = (
                    f"MongoDB: update document _id={mid!r} only."
                    if mid
                    else "MongoDB: update rows whose filepath matches source path."
                )
                lines.append(
                    f"   -> move: {a} -> {b} | source exists: {exists} | {mongo_note}"
                )
            elif act == "remove_file":
                p_raw = step.get("path")
                mid = step.get("mongo_id") or step.get("mongo_object_id")
                src_note = ""
                if (not p_raw or not str(p_raw).strip()) and mid:
                    try:
                        fp = _filepath_from_mongo_id(str(mid))
                        src_note = f"path resolved from index: {fp} | "
                        p = _resolve_under_root(fp)
                    except Exception as ex:
                        lines.append(f"   !! {ex}")
                        p = None
                else:
                    p = _resolve_under_root(str(p_raw or ""))
                if p is not None:
                    lines.append(
                        f"   -> move file to system Trash: {p} | {src_note}exists: {p.is_file()} | "
                        f"MongoDB: {'delete _id=' + repr(mid) if mid else 'delete by filepath'} | "
                        f"undo: macOS/Linux if Trash path is recorded"
                    )
            elif act == "add_file":
                p = _resolve_under_root(str(step.get("path", "")))
                lines.append(
                    f"   -> index file (Gemini embed + insert): {p} | exists: {p.is_file()}"
                )
            elif act == "remove_folder":
                p = _resolve_under_root(str(step.get("path", "")))
                root = _fs_root().resolve()
                is_root = p.resolve() == root
                lines.append(
                    f"   -> move folder to system Trash: {p} | is_dir: {p.is_dir()} | "
                    f"MongoDB: delete all rows with filepath under this folder | "
                    f"refused if project root: {is_root} | undo: no"
                )
        except Exception as ex:
            lines.append(f"   !! {ex}")
    return "\n".join(lines)


@tool
def preview_plan(plan_json: str) -> str:
    """Show what a JSON plan would do without changing disk or MongoDB.

    ``execute_plan`` already runs this same preview **before** executing; use ``preview_plan`` alone
    when you only want to inspect a plan without running it.

    ``plan_json`` is a JSON array of steps, e.g.:
    [{"action": "create_folder", "path": "notes/2026"},
     {"action": "add_file", "path": "notes/2026/report.pdf", "description": "Q4 report"},
     {"action": "remove_file", "path": "old/tmp.txt"},
     {"action": "remove_file", "path": "dup.pdf", "mongo_id": "674a1b2c3d4e5f6789012345"},
     {"action": "move_file", "from": "draft.txt", "to": "notes/2026/draft.txt",
      "mongo_id": "674a1b2c3d4e5f6789012345"},
     {"action": "remove_folder", "path": "notes/old_batch"}]

    Use ``mongo_id`` on ``move_file`` / ``remove_file`` when targeting one row. Paths are under the project root (``YHACKS_FS_ROOT`` or cwd)."""
    try:
        steps = _parse_plan_json(plan_json)
    except Exception as e:
        return f"Invalid plan JSON: {e}"
    return _format_plan_preview(steps)


@tool
def execute_plan(plan_json: str, dry_run: bool = False) -> str:
    """Execute a JSON plan (see ``preview_plan``).

    **Always runs the same preview as ``preview_plan`` first**, then executes (unless ``dry_run=True``).

    ``remove_file`` uses the system Trash; on macOS/Linux an undo batch may restore from Trash and re-index.
    ``remove_folder`` is not recorded for undo. ``add_file`` undo removes only the new MongoDB row (file stays).
    For ``move_file``, set ``mongo_id`` to target one document by ``_id``; else match by ``filepath``.

    Set ``dry_run=True`` to only return the preview with no disk or MongoDB changes.

    Appends one undo batch for the whole plan (reversed on ``undo_last_action``). Stops at first error."""
    try:
        steps = _parse_plan_json(plan_json)
    except Exception as e:
        return f"Invalid plan JSON: {e}"
    preview = _format_plan_preview(steps)
    header = "=== Plan preview (before execution) ===\n" + preview
    if dry_run:
        return header + "\n\n=== dry_run=True — no execution ==="

    log: list[str] = []
    per_step_frags: list[list[dict[str, Any]]] = []
    for step in steps:
        try:
            msg, frags = _execute_one_step(step)
        except Exception as e:
            log.append(f"FAILED on step {step!r}: {e}")
            return header + "\n\n=== Execution ===\nStopped.\n" + "\n".join(log)
        log.append(msg)
        per_step_frags.append(frags)
    combined: list[dict[str, Any]] = []
    for fr in reversed(per_step_frags):
        combined.extend(fr)
    _undo_stack.append(combined)
    return (
        header
        + "\n\n=== Execution ===\nDone.\n"
        + "\n".join(log)
        + f"\nUndo batch recorded ({len(combined)} atomic undo(s)). "
        f"Stack depth: {len(_undo_stack)}."
    )


@tool
def undo_last_action() -> str:
    """Undo the most recent batch from ``execute_plan`` or ``trash_file`` (last batch first, LIFO)."""
    if not _undo_stack:
        return "Nothing to undo."
    batch = _undo_stack.pop()
    errors: list[str] = []
    for u in batch:
        try:
            _run_undo_step(u)
        except Exception as e:
            errors.append(f"{u.get('op')}: {e}")
    if errors:
        return "Undo finished with errors:\n" + "\n".join(errors)
    return f"Undid {len(batch)} atomic step(s). Remaining undo batches: {len(_undo_stack)}."


AGENT_TOOLS = [
    semantic_file_search,
    ask_about_files,
    trash_file,
    preview_plan,
    execute_plan,
    undo_last_action,
]


# ---------------------------------------------------------------------------
# Graph + factory
# ---------------------------------------------------------------------------


def _make_call_model(llm_with_tools):
    def call_model(state: MessagesState) -> dict:
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    return call_model


def build_graph(llm_with_tools):
    graph = StateGraph(MessagesState)
    graph.add_node("agent", _make_call_model(llm_with_tools))
    graph.add_node("tools", ToolNode(AGENT_TOOLS))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", "__end__": END},
    )
    graph.add_edge("tools", "agent")
    return graph.compile()


def create_chat_app(model: str | None = None):
    """Build compiled LangGraph app: ``app.invoke({\"messages\": [...]})``."""
    api_key = load_google_api_key()
    name = model or os.environ.get("AGENT_MODEL", "gemini-2.5-flash")
    llm = ChatGoogleGenerativeAI(
        model=name,
        temperature=0,
        google_api_key=api_key,
    )
    return build_graph(llm.bind_tools(AGENT_TOOLS))


def last_assistant_reply(messages: list) -> str:
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            t = getattr(m, "text", None) or ""
            if t:
                return str(t)
    return ""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _repl() -> None:
    app = create_chat_app()
    messages: list = [SystemMessage(content=DEFAULT_AGENT_SYSTEM)]
    print("Agent ready (empty line to exit). Working directory should be backend/ for imports.")
    while True:
        try:
            line = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            break
        messages.append(HumanMessage(content=line))
        out = app.invoke({"messages": messages})
        messages = out["messages"]
        print("Agent:", last_assistant_reply(messages) or "(no text reply)")


def main() -> None:
    _repl()


if __name__ == "__main__":
    main()
