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
- ``{"action": "remove_file", "path": "old.txt"}`` — deletes the file on disk and removes matching
  MongoDB rows (optional ``mongo_id`` to delete one document only). **Not undoable** (no file restore).
- ``{"action": "add_file", "path": "doc.pdf", "description": "optional"}`` — indexes an existing file
  with ``ingest_file_to_db`` (embedding + insert). Undo removes the new MongoDB row only (file stays).
- ``{"action": "remove_folder", "path": "relative/dir"}`` — recursively deletes the directory on disk
  and removes MongoDB rows whose ``filepath`` lies under that folder. **Not undoable.** Cannot remove
  the project root.

    pip install langchain-core langchain-google-genai langgraph
    python agent.py

**Plans:** ``execute_plan`` always runs the same preview as ``preview_plan`` before executing.
Use ``dry_run=True`` on ``execute_plan`` for preview only. Other tools (e.g. ``semantic_file_search``)
do not auto-preview.

Add tools: define ``@tool``, append to ``AGENT_TOOLS``, document parameters in the docstring.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

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

DEFAULT_AGENT_SYSTEM = (
    "You help with file organization and MongoDB vector indexing. "
    "For JSON plans, call execute_plan: it shows a plan preview before running steps. "
    "Use execute_plan(plan_json, dry_run=True) to preview without side effects. "
    "preview_plan is optional; it duplicates what execute_plan shows first."
)


# ---------------------------------------------------------------------------
# Filesystem root + undo stack (used by create_folder / plan / undo)
# ---------------------------------------------------------------------------

# All FS tools confine paths under this directory (set YHACKS_FS_ROOT to your project root).
_undo_stack: list[list[dict[str, Any]]] = []


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


def _normalize_mongo_id(raw: str | None) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip()
    return s or None


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
        extra = f" mongo_id={mid!r}" if mid else ""
        return f"{i}. remove_file: {step.get('path', '?')}{extra}"
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


def _apply_remove_file(path_str: str, mongo_id: str | None = None) -> tuple[str, list[dict[str, Any]]]:
    """Delete file on disk and MongoDB rows. Undo not supported (returns empty frags)."""
    p = _resolve_under_root(path_str)
    if not p.is_file():
        raise FileNotFoundError(f"Not a file or missing: {p}")
    abs_path = str(p.resolve())
    mid = _normalize_mongo_id(mongo_id)
    deleted = 0
    if mid:
        try:
            ObjectId(mid)
        except Exception as e:
            raise ValueError(f"Invalid mongo_id: {mongo_id!r}") from e
        r = remove_mod.collection.delete_one({"_id": ObjectId(mid)})
        deleted = r.deleted_count
    else:
        r = remove_mod.collection.delete_many({"filepath": abs_path})
        deleted = r.deleted_count
    p.unlink()
    msg = f"Removed file {abs_path}; MongoDB document(s) deleted: {deleted}"
    return msg, []


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
    """Recursively delete a directory and MongoDB rows for files under it. Not undoable."""
    p = _resolve_under_root(path_str)
    root = _fs_root().resolve()
    rp = p.resolve()
    if not rp.is_dir():
        raise NotADirectoryError(f"Not a directory or missing: {rp}")
    if rp == root:
        raise ValueError("Refusing to remove the project root directory.")
    n_mongo = _mongo_delete_files_under_folder(str(rp))
    shutil.rmtree(rp)
    msg = f"Removed folder {rp} (recursive); MongoDB document(s) deleted: {n_mongo}"
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
        if not pth:
            raise ValueError("remove_file requires 'path'.")
        mid = step.get("mongo_id") or step.get("mongo_object_id")
        return _apply_remove_file(str(pth), mongo_id=str(mid) if mid else None)
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
                p = _resolve_under_root(str(step.get("path", "")))
                mid = step.get("mongo_id") or step.get("mongo_object_id")
                lines.append(
                    f"   -> delete file: {p} | exists: {p.is_file()} | "
                    f"MongoDB: {'delete _id=' + repr(mid) if mid else 'delete by filepath'}"
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
                    f"   -> rmtree: {p} | is_dir: {p.is_dir()} | "
                    f"MongoDB: delete all rows with filepath under this folder | "
                    f"refused if project root: {is_root}"
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

    ``remove_file`` / ``remove_folder`` cannot be reversed via undo (disk deleted). ``add_file`` undo removes only the
    new MongoDB row. For ``move_file``, set ``mongo_id`` to target one document by ``_id``; else match by ``filepath``.

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
    """Undo the most recent ``execute_plan`` or ``create_folder`` (last batch first, LIFO)."""
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
    preview_plan,
    execute_plan,
    undo_last_action,
]


# ---------------------------------------------------------------------------
# Graph + factory
# ---------------------------------------------------------------------------


def load_google_api_key() -> str:
    raw = os.environ.get("GEMINI_API_KEY")
    key = raw.strip() if raw else ""
    if not key:
        raise ValueError(
            "Set GEMINI_API_KEY in the environment before running the agent."
        )
    if "\n" in raw or "\r" in raw:
        raise ValueError("API key must be a single line with no newlines.")
    return key


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
