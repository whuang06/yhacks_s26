"""
Shared LangGraph + LangChain agent (tools, graph, helpers) for CLI and HTTP server.
"""

from __future__ import annotations

from env_bootstrap import load_project_env

load_project_env()

import os
import re
from datetime import datetime
from pathlib import Path

from rapidfuzz import fuzz

# Skip heavy / irrelevant trees when resolving fuzzy rename hints
_SKIP_DIR_NAMES = frozenset(
    {
        ".git",
        "node_modules",
        ".venv",
        "venv",
        "__pycache__",
        "target",
        "dist",
        "build",
        ".cursor",
    }
)

# Fuzzy / “semantic” name matching (token overlap + similarity; not ML embeddings)
_FUZZY_MIN_SCORE = 72
_FUZZY_MIN_SCORE_SHORT = 84  # single-token hints: avoid partial_ratio false positives
_FUZZY_AMBIGUITY_GAP = 9

_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "to",
        "for",
        "and",
        "or",
        "file",
        "files",
        "folder",
        "this",
        "that",
        "my",
        "me",
        "please",
        "rename",
        "renamed",
        "move",
        "into",
        "called",
    }
)

from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from file_tools import FILE_TOOLS
from db_sync_tools import DB_SYNC_TOOLS

# (original_absolute_path, new_absolute_path) after each successful rename_file
_rename_stack: list[tuple[str, str]] = []


def _resolve_under_cwd(user_path: str) -> Path:
    """Resolve a path and require it stays under the current working directory."""
    base = Path.cwd().resolve()
    p = Path(user_path).expanduser()
    full = p.resolve() if p.is_absolute() else (base / p).resolve()
    try:
        full.relative_to(base)
    except ValueError as e:
        raise ValueError("Paths must be inside the current working directory.") from e
    return full


def _normalize_hint_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]+", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _keyword_only_hint(normalized: str) -> str:
    words = [w for w in normalized.split() if w not in _STOPWORDS and len(w) > 1]
    return " ".join(words) if words else normalized


def _path_name_similarity(hint_raw: str, p: Path, base: Path) -> int:
    """How well ``hint_raw`` matches this path’s name, stem, or relative path (0–100)."""
    raw = hint_raw.strip().strip('`"\'')
    if not raw:
        return 0
    full_norm = _normalize_hint_text(raw)
    key_norm = _keyword_only_hint(full_norm)
    rel = str(p.relative_to(base))
    candidates = (
        p.name,
        p.stem,
        rel,
        rel.replace("/", " ").replace("\\", " "),
    )
    best = 0
    for hint_variant in (raw.lower(), full_norm, key_norm):
        if not hint_variant:
            continue
        for pv in candidates:
            pl = pv.lower()
            best = max(
                best,
                fuzz.WRatio(hint_variant, pl),
                fuzz.token_set_ratio(hint_variant, pl),
                fuzz.partial_token_sort_ratio(hint_variant, pl),
                fuzz.partial_ratio(hint_variant, pl),
            )
    return int(best)


def _fuzzy_min_for_hint(hint_raw: str) -> int:
    """Stricter cutoff for a single keyword so ``image`` does not match ``agent``."""
    key = _keyword_only_hint(_normalize_hint_text(hint_raw))
    tokens = [t for t in key.split() if len(t) > 1]
    if len(tokens) <= 1:
        return _FUZZY_MIN_SCORE_SHORT
    return _FUZZY_MIN_SCORE


def _find_sources_for_hint(hint: str) -> list[Path]:
    """Find paths under cwd matching ``hint`` (exact, then stem, then fuzzy / semantic).

    1. Resolved path if it exists.
    2. Case-insensitive basename match.
    3. Case-insensitive stem match (``image`` -> ``photo/image.png``).
    4. Fuzzy match on name, stem, and relative path (natural phrases like
       “hacker guide” -> ``YHack26 Hacker Guide.pdf``).

    When fuzzy matching, if several paths score within ``_FUZZY_AMBIGUITY_GAP`` of
    the best score, all are returned so the caller can ask for disambiguation.
    """
    raw = hint.strip().strip('`"\'')
    if not raw:
        return []

    base = Path.cwd().resolve()
    try:
        resolved = _resolve_under_cwd(raw)
        if resolved.exists():
            return [resolved]
    except ValueError:
        pass

    last = Path(raw).name
    want_name = last.lower()
    want_stem = Path(last).stem.lower()

    exact: list[Path] = []
    by_stem: list[Path] = []
    all_paths: list[Path] = []

    for p in base.rglob("*"):
        if not p.exists():
            continue
        if any(part in _SKIP_DIR_NAMES for part in p.parts):
            continue
        try:
            p.relative_to(base)
        except ValueError:
            continue
        all_paths.append(p)
        name_l = p.name.lower()
        stem_l = p.stem.lower()
        if name_l == want_name:
            exact.append(p)
        elif stem_l == want_stem:
            by_stem.append(p)

    if exact:
        return sorted({p.resolve() for p in exact}, key=lambda x: str(x))
    if by_stem:
        return sorted({p.resolve() for p in by_stem}, key=lambda x: str(x))

    min_fuzzy = _fuzzy_min_for_hint(raw)
    scored: list[tuple[int, Path]] = []
    for p in all_paths:
        s = _path_name_similarity(raw, p, base)
        if s >= min_fuzzy:
            scored.append((s, p.resolve()))

    if not scored:
        return []

    scored.sort(key=lambda x: (-x[0], str(x[1])))
    best = scored[0][0]
    band = [path for score, path in scored if score >= best - _FUZZY_AMBIGUITY_GAP]
    return sorted(set(band), key=str)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
def rename_file(source_path: str, destination_path: str) -> str:
    """Rename or move a file or directory under the current working directory.

    ``source_path`` may be a natural description: exact basename, stem (``image``
    -> ``image.png``), or a loose phrase that closely matches the file name
    (e.g. ``hacker guide pdf`` -> ``YHack26 Hacker Guide.pdf``). If several paths
    match, the tool lists them so the user can narrow the hint.

    ``destination_path`` may use subfolders; missing parent directories are created.
    If the source is a file with an extension and the destination has no extension,
    the same extension is kept (e.g. rename ``image`` to ``Nathan`` -> ``Nathan.png``).

    Fails if the destination already exists. Each successful call can be reversed once
    by ``undo_last_file_rename`` (most recent first).
    """
    try:
        dst = _resolve_under_cwd(destination_path)
    except ValueError as e:
        return str(e)

    candidates = _find_sources_for_hint(source_path)
    if not candidates:
        return (
            f"No file or folder under the project matches {source_path!r}. "
            "Try a shorter name (e.g. stem without extension) or a more specific basename."
        )
    if len(candidates) > 1:
        lines = "\n".join(f"  - {c.relative_to(Path.cwd().resolve())}" for c in candidates[:25])
        more = f"\n  ... and {len(candidates) - 25} more" if len(candidates) > 25 else ""
        return (
            f"Multiple matches for {source_path!r}; be more specific:\n{lines}{more}"
        )

    src = candidates[0]
    if dst.exists():
        return f"Destination already exists: {dst}"

    if src.is_file() and not dst.suffix and src.suffix:
        dst = dst.with_suffix(src.suffix)

    if dst.exists():
        return f"Destination already exists: {dst}"

    dst.parent.mkdir(parents=True, exist_ok=True)
    src_abs, dst_abs = str(src), str(dst)
    try:
        src.rename(dst)
    except OSError as e:
        return f"Rename failed: {e}"
    _rename_stack.append((src_abs, dst_abs))
    return f"Renamed: {src_abs} -> {dst_abs}"


@tool
def undo_last_file_rename() -> str:
    """Undo the most recent successful ``rename_file`` (LIFO). Safe to call repeatedly."""
    if not _rename_stack:
        return "Nothing to undo (no prior rename_file in this process)."
    old_abs, new_abs = _rename_stack.pop()
    new_p, old_p = Path(new_abs), Path(old_abs)
    if not new_p.exists():
        _rename_stack.append((old_abs, new_abs))
        return f"Cannot undo: {new_p} does not exist."
    if old_p.exists():
        _rename_stack.append((old_abs, new_abs))
        return f"Cannot undo: original path already exists: {old_p}"
    try:
        new_p.rename(old_p)
    except OSError as e:
        _rename_stack.append((old_abs, new_abs))
        return f"Undo failed: {e}"
    return f"Undid rename: {new_p} -> {old_p}"


@tool
def add_file_embedding_to_database(rel_path: str, description: str = "") -> str:
    """Index a project file for semantic image/document search.

    Calls Gemini ``gemini-embedding-2-preview`` on the file bytes (multimodal:
    PDF, images, text) and upserts the vector into MongoDB Atlas under
    ``rel_path``. Requires ``MONGO_URI`` and a READY vector index (see
    ``scripts/create_vector_index.py``).

    ``rel_path`` is relative to the project root (e.g. ``timed_outputs/photo.png``).
    Optional ``description`` is prepended to the embedding input for extra context.
    """
    try:
        from semantic_backend import repository as sem
    except ImportError as e:
        return f"Semantic stack import failed (install deps): {e}"
    try:
        ok, msg = sem.upsert_file_embedding(Path.cwd().resolve(), rel_path, description or None)
    except Exception as e:
        return f"Semantic index error: {sem.format_mongo_connection_error(e)}"
    return msg if ok else msg


@tool
def search_semantic_files(query: str) -> str:
    """Search files with natural language (images + documents).

    Uses MongoDB Atlas vector search (Gemini embeddings + RRF), auto-indexes
    ``timed_outputs/``, filters out ``desktop/`` and ``node_modules`` hits, and
    merges cosine-ranked **images and PDFs** from ``timed_outputs`` (nested paths
    and random names OK). Queries mentioning ``pdf`` get an extra PDF rank pass in
    fusion so PDFs are not drowned out by images. If needed, falls back to filename
    similarity on disk.
    """
    try:
        from semantic_backend import repository as sem
    except ImportError as e:
        return f"Semantic stack import failed: {e}"
    try:
        hits = sem.search_semantic_top5(query, Path.cwd().resolve())
    except Exception as e:
        hint = sem.format_mongo_connection_error(e)
        return f"Semantic search error: {hint}"
    if not hits:
        from semantic_backend.embeddings import diagnose_semantic_embeddings

        base = (
            "No results: nothing matched in the vector index and no project files "
            "looked similar to your query (try different keywords)."
        )
        diag = diagnose_semantic_embeddings(Path.cwd().resolve())
        if diag:
            return f"{base}\n\nDiagnosis: {diag}"
        return base
    from_fallback = hits[0].get("source") == "filename_match"
    header = (
        "Top matches (local filename / path — index empty or no vector hits):\n"
        if from_fallback
        else "Top 5 (vector + RRF):\n"
    )
    lines = []
    for i, h in enumerate(hits, 1):
        rp = h.get("rel_path") or h.get("filepath") or "?"
        ft = h.get("file_type") or "?"
        sc = h.get("rrf_score", 0)
        lines.append(f"{i}. {rp}  (type={ft}, rrf={sc})")
    return header + "\n".join(lines)


@tool
def remove_file_embedding_from_database(rel_path: str) -> str:
    """Remove the vector document for a project file from MongoDB (e.g. after the file was deleted).

    Matches on stored ``rel_path`` (relative to project root, same as indexing).
    """
    try:
        from semantic_backend import repository as sem
    except ImportError as e:
        return f"Semantic stack import failed: {e}"
    try:
        n, msg = sem.remove_embeddings_for_rel_path(rel_path)
    except Exception as e:
        return f"Remove embedding error: {sem.format_mongo_connection_error(e)}"
    return msg if n else f"No embedding row for rel_path={rel_path!r}."


@tool
def open_file(rel_path: str) -> str:
    """Preview a project file’s contents for the user (path relative to project root).

    Returns a UTF-8 text excerpt for text-like files. For PDFs and binary files,
    returns a short notice and the path so the desktop preview pane can open it.
    """
    try:
        p = _resolve_under_cwd(rel_path)
    except ValueError as e:
        return str(e)
    if not p.exists():
        return f"Not found: {rel_path}"
    if p.is_dir():
        return f"Is a directory: {rel_path}"
    suf = p.suffix.lower()
    if suf == ".pdf":
        return (
            f"PDF at `{rel_path}` — open the Preview tab in the app for full view "
            f"({p.stat().st_size} bytes)."
        )
    image_sfx = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}
    if suf in image_sfx:
        return (
            f"Image at `{rel_path}` — use the app Preview tab ({p.stat().st_size} bytes)."
        )
    data = p.read_bytes()
    if len(data) > 400_000:
        return f"File too large to inline ({len(data)} bytes): {rel_path}"
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = data.decode("latin-1")
        except Exception:
            return f"Binary file; cannot show as text: {rel_path}"
    printable = sum(1 for c in text[:4000] if c.isprintable() or c in "\n\r\t")
    if len(text) > 200 and printable / min(len(text), 4000) < 0.82:
        return f"Mostly non-text bytes; use Preview for: {rel_path}"
    cap = 12_000
    body = text if len(text) <= cap else text[:cap] + "\n\n… [truncated]"
    return f"--- {rel_path} ({len(text)} chars) ---\n{body}"


TOOLS = [
    add_numbers,
    multiply_numbers,
    create_timestamp_named_file,
    rename_file,
    undo_last_file_rename,
    add_file_embedding_to_database,
    search_semantic_files,
    remove_file_embedding_from_database,
    open_file,
    *FILE_TOOLS,
    *DB_SYNC_TOOLS,
]


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


def make_call_model(llm_with_tools):
    def call_model(state: MessagesState) -> dict:
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    return call_model


def build_graph(llm_with_tools):
    graph = StateGraph(MessagesState)
    graph.add_node("agent", make_call_model(llm_with_tools))
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", "__end__": END},
    )
    graph.add_edge("tools", "agent")
    return graph.compile()


def last_assistant_reply(messages: list) -> str:
    for m in reversed(messages):
        if isinstance(m, AIMessage) and m.text:
            return str(m.text)
    return ""


# ---------------------------------------------------------------------------
# API key
# ---------------------------------------------------------------------------


def load_google_api_key() -> str:
    raw = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or ""
    api_key = raw.strip()
    if not api_key:
        raise ValueError(
            "No Gemini API key found. Add GEMINI_API_KEY=... to desktop/.env (project "
            "test_langchain/desktop/.env), or export it in the same shell that runs "
            "`npm run dev`. Putting the key only in .venv/bin/activate does not apply "
            "when Tauri starts uvicorn unless that shell sourced activate first. Remove "
            "any empty GEMINI_API_KEY= line from .env."
        )
    if "\n" in raw or "\r" in raw or "export " in api_key.lower():
        raise ValueError(
            "API key env var must be ONLY the key string, not shell commands or multiple lines."
        )
    return api_key


def create_chat_app():
    """Compiled LangGraph app (invoke with {\"messages\": [...]})."""
    api_key = load_google_api_key()
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=api_key,
    )
    return build_graph(llm.bind_tools(TOOLS))
