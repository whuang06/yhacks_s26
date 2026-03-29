"""
FastAPI backend for the CopperGolem Tauri desktop UI.

Wraps the existing yhacks_s26/backend/ agent and vector-search modules
without modifying them. Run from the front_end directory:

    python -m uvicorn server:app --host 127.0.0.1 --port 8765
"""

from __future__ import annotations

import base64
import os
import re
import sys
import uuid
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: load .env, then put backend/ on sys.path so its imports resolve.
# ---------------------------------------------------------------------------

from env_bootstrap import load_project_env

load_project_env()

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BACKEND_DIR = _REPO_ROOT / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Lazy imports for the backend.  input_to_embedding.py creates a Gemini
# client at *module level*, so importing the agent chain at top-level
# crashes the whole server when GEMINI_API_KEY is missing.  By deferring,
# health / session / browse / preview still work; the agent fails only on
# endpoints that actually need it.
# ---------------------------------------------------------------------------

_agent_mod = None


def _import_agent():
    global _agent_mod
    if _agent_mod is None:
        import agent as _mod  # noqa: E402

        _agent_mod = _mod
    return _agent_mod


def _langchain_messages():
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    return AIMessage, HumanMessage, SystemMessage, ToolMessage


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

_sessions: dict[str, list] = {}
_graph = None

MAX_PREVIEW_BYTES = 20 * 1024 * 1024  # 20 MiB (images, text, etc.)
MAX_PDF_PREVIEW_BYTES = 40 * 1024 * 1024

app = FastAPI(title="CopperGolem API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_graph():
    global _graph
    if _graph is None:
        os.environ.setdefault("YHACKS_FS_ROOT", str(_REPO_ROOT.resolve()))
        mod = _import_agent()
        _graph = mod.create_chat_app()
    return _graph


def _content_roots() -> list[Path]:
    roots: list[Path] = []
    repo = _REPO_ROOT.resolve()
    roots.append(repo)

    if not (os.environ.get("COPPERGOLEM_NO_PARENT_ROOT") or "").strip():
        parent = repo.parent
        if parent.is_dir() and parent.resolve() not in roots:
            roots.append(parent.resolve())

    extra = os.environ.get("COPPERGOLEM_EXTRA_ROOTS", "")
    for token in re.split(r"[|;:]", extra):
        raw = token.strip()
        if not raw:
            continue
        p = Path(raw).expanduser().resolve()
        if p.is_dir() and p not in roots:
            roots.append(p)

    return roots


def _safe_resolve_under_roots(rel: str) -> Path:
    rel_norm = (rel or "").strip().replace("\\", "/").lstrip("/")
    if rel_norm in ("", "."):
        return _REPO_ROOT.resolve()
    if any(part == ".." for part in rel_norm.split("/")):
        raise HTTPException(status_code=400, detail="Invalid path")

    candidates: list[tuple[Path, Path]] = []
    for root in _content_roots():
        r = root.resolve()
        full = (r / rel_norm).resolve()
        try:
            full.relative_to(r)
        except ValueError:
            continue
        candidates.append((full, r))

    if not candidates:
        raise HTTPException(
            status_code=403,
            detail="Path outside allowed workspace roots",
        ) from None

    existing = [(f, rt) for f, rt in candidates if f.exists()]
    pick_from = existing if existing else candidates
    pick_from.sort(key=lambda fr: len(str(fr[1])), reverse=True)
    return pick_from[0][0]


def _rel_for_api(p: Path) -> str:
    pr = p.resolve()
    roots = sorted(_content_roots(), key=lambda r: len(str(r.resolve())), reverse=True)
    for root in roots:
        try:
            return str(pr.relative_to(root.resolve())).replace("\\", "/")
        except ValueError:
            continue
    return str(pr)


def _path_allowed(p: Path) -> bool:
    pr = p.resolve()
    for root in _content_roots():
        r = root.resolve()
        try:
            pr.relative_to(r)
            return True
        except ValueError:
            continue
    return False


# ---------------------------------------------------------------------------
# Chat models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    found_files: list[str] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    is_plan_proposal: bool = Field(
        default=False,
        description="True when the agent previewed a plan (dry_run) and awaits user confirmation",
    )


# ---------------------------------------------------------------------------
# Parse tool results from message history
# ---------------------------------------------------------------------------

_FOUND_PATH_RE = re.compile(r"^\s+path:\s+(.+)$", re.MULTILINE)
_FOUND_FILE_RE_WITH_ID = re.compile(
    r"^\s*\d+\.\s+score=[\d.]+\s+\|\s+_id=\S+\s+\|\s+(.+?)\s+\|", re.MULTILINE
)
_FOUND_FILE_RE_LEGACY = re.compile(
    r"^\s*\d+\.\s+score=[\d.]+\s+\|\s+(.+?)\s+\|", re.MULTILINE
)


def _messages_after_last_human(messages: list) -> list:
    AIMessage, HumanMessage, SystemMessage, ToolMessage = _langchain_messages()
    last_human_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            last_human_idx = i
            break
    return messages[last_human_idx + 1 :] if last_human_idx >= 0 else messages


def extract_tools_used(messages: list) -> list[str]:
    AIMessage, HumanMessage, SystemMessage, ToolMessage = _langchain_messages()
    recent = _messages_after_last_human(messages)
    out: list[str] = []
    seen: set[str] = set()
    for m in recent:
        if not isinstance(m, ToolMessage):
            continue
        name = getattr(m, "name", None)
        if name and name not in seen:
            seen.add(name)
            out.append(str(name))
    return out


def extract_found_files(messages: list) -> list[str]:
    AIMessage, HumanMessage, SystemMessage, ToolMessage = _langchain_messages()
    recent = _messages_after_last_human(messages)

    out: list[str] = []
    seen: set[str] = set()
    for m in recent:
        if not isinstance(m, ToolMessage):
            continue
        content = m.content if isinstance(m.content, str) else str(m.content)
        if "score=" not in content:
            continue
        for mo in _FOUND_PATH_RE.finditer(content):
            p = mo.group(1).strip()
            if p and p not in seen:
                seen.add(p)
                out.append(p)
        if not out:
            for rx in (_FOUND_FILE_RE_WITH_ID, _FOUND_FILE_RE_LEGACY):
                for mo in rx.finditer(content):
                    fname = mo.group(1).strip()
                    if fname and fname not in seen:
                        seen.add(fname)
                        out.append(fname)
    return out


def _detect_plan_proposal(messages: list, tools_used: list[str]) -> bool:
    """Return True when the agent ran a plan preview (dry_run) but hasn't executed yet."""
    plan_tools = {"preview_plan", "execute_plan"}
    if not plan_tools.intersection(tools_used):
        return False
    AIMessage, HumanMessage, SystemMessage, ToolMessage = _langchain_messages()
    recent = _messages_after_last_human(messages)
    for m in recent:
        if not isinstance(m, ToolMessage):
            continue
        content = m.content if isinstance(m.content, str) else str(m.content)
        if "dry_run" in content.lower() or "Plan preview" in content:
            return True
        if "=== Execution ===" in content and "Done." in content:
            return False
    return False


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    mongo_ok = False
    try:
        from pymongo import MongoClient

        uri = os.environ.get("MONGO_URI", "")
        if uri:
            c = MongoClient(uri, serverSelectionTimeoutMS=3000)
            c.admin.command("ping")
            mongo_ok = True
    except Exception:
        pass
    return {
        "ok": True,
        "mongo": mongo_ok,
        "gemini_key_set": bool((os.environ.get("GEMINI_API_KEY") or "").strip()),
    }


@app.post("/api/session/new")
def new_session():
    sid = str(uuid.uuid4())
    _sessions[sid] = []
    return {"session_id": sid}


@app.get("/api/agent/tools")
def agent_tools():
    """Expose tool names/descriptions so the UI can show capabilities."""
    try:
        mod = _import_agent()
        tools_out: list[dict[str, str]] = []
        for t in mod.AGENT_TOOLS:
            desc = (getattr(t, "description", None) or "").strip()
            first_line = desc.split("\n", 1)[0].strip() if desc else ""
            tools_out.append({"name": getattr(t, "name", ""), "description": first_line})
        return {
            "tools": tools_out,
            "fs_root": os.environ.get("YHACKS_FS_ROOT", "") or str(_REPO_ROOT.resolve()),
            "system_prompt": getattr(mod, "DEFAULT_AGENT_SYSTEM", ""),
        }
    except Exception:
        return {
            "tools": [
                {"name": "semantic_file_search", "description": "Search files by meaning (vector similarity)"},
                {"name": "preview_plan", "description": "Preview a file-management plan without executing"},
                {"name": "execute_plan", "description": "Execute a file-management plan (create, move, remove, index)"},
                {"name": "undo_last_action", "description": "Undo the last executed plan"},
            ],
            "fs_root": os.environ.get("YHACKS_FS_ROOT", "") or str(_REPO_ROOT.resolve()),
            "system_prompt": "",
        }


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    AIMessage, HumanMessage, SystemMessage, ToolMessage = _langchain_messages()
    mod = _import_agent()

    sid = req.session_id or str(uuid.uuid4())
    if sid not in _sessions:
        _sessions[sid] = []

    messages = _sessions[sid]

    if not messages:
        base = getattr(mod, "DEFAULT_AGENT_SYSTEM", "") or ""
        sys_text = (
            base.strip()
            + "\n\nIMPORTANT RULES:"
            "\n1. When the user asks you to organize, move, rename, "
            "create, remove, or index files, ALWAYS call execute_plan with "
            "dry_run=True FIRST. Show the preview and ask the user to confirm "
            "before executing. Only call execute_plan with dry_run=False after "
            "the user explicitly confirms. Never execute destructive actions "
            "without confirmation."
            "\n2. For semantic_file_search: be VERY selective about results. "
            "Use k=3 when the user wants ONE specific file. Use k=8 for broader "
            "category searches. After getting results, CRITICALLY evaluate them: "
            "look at the scores and look for a SCORE GAP — a meaningful drop "
            "between consecutive results. Results that cluster near the top "
            "are likely relevant; results after a noticeable score drop are noise. "
            "For example if scores are 0.69, 0.68, 0.65, 0.64, 0.64 — the top 2 "
            "are clearly better and the rest are noise. ONLY report results that "
            "are genuinely relevant to the query. It is much better to return "
            "1-2 highly relevant files than 10 mediocre ones. If none of the "
            "scores are notably higher than the rest, tell the user nothing "
            "strongly matched."
        )
        messages.append(SystemMessage(content=sys_text))

    messages.append(HumanMessage(content=req.message.strip()))
    try:
        final = get_graph().invoke({"messages": messages})
    except Exception as e:
        messages.pop()
        raise HTTPException(status_code=500, detail=str(e)) from e

    messages[:] = final["messages"]
    reply = mod.last_assistant_reply(messages)
    found = extract_found_files(messages)
    used = extract_tools_used(messages)

    is_plan = _detect_plan_proposal(messages, used)

    return ChatResponse(
        session_id=sid,
        reply=reply or "(no text reply)",
        found_files=found,
        tools_used=used,
        is_plan_proposal=is_plan,
    )


# ---------------------------------------------------------------------------
# File browsing & preview
# ---------------------------------------------------------------------------


@app.get("/api/browse")
def browse(rel_path: str | None = Query(default=None)):
    effective = rel_path.strip().replace("\\", "/").lstrip("/") if rel_path else ""
    p = _safe_resolve_under_roots(effective)
    if not p.exists():
        return {"path": _rel_for_api(p), "entries": [], "missing": True}
    if p.is_file():
        return {"path": _rel_for_api(p), "entries": [], "is_file": True}
    entries = []
    for child in sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
        if child.name.startswith("."):
            continue
        if not _path_allowed(child):
            continue
        st = child.stat()
        entries.append(
            {
                "name": child.name,
                "path": _rel_for_api(child),
                "is_dir": child.is_dir(),
                "size": None if child.is_dir() else st.st_size,
            }
        )
    return {"path": _rel_for_api(p), "entries": entries}


def _find_file_by_name(filename: str) -> Path | None:
    """Walk content roots looking for *filename*. Searches extra roots first."""
    extra = os.environ.get("COPPERGOLEM_EXTRA_ROOTS", "")
    extra_paths = set()
    for token in re.split(r"[|;:]", extra):
        raw = token.strip()
        if raw:
            extra_paths.add(Path(raw).expanduser().resolve())

    roots = _content_roots()
    ordered = sorted(roots, key=lambda r: r.resolve() not in extra_paths)

    for root in ordered:
        for dirpath, dirs, files in os.walk(root.resolve()):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            if filename in files:
                candidate = Path(dirpath) / filename
                if candidate.is_file() and _path_allowed(candidate):
                    return candidate.resolve()
    return None


@app.get("/api/file/preview")
def file_preview(rel_path: str = Query(..., min_length=1)):
    raw = rel_path.strip()

    if os.path.isabs(raw):
        p = Path(raw).expanduser().resolve()
        if not _path_allowed(p) or not p.exists() or not p.is_file():
            resolved = None
            parts = Path(raw).parts[1:]
            for length in range(len(parts), 0, -1):
                candidate_rel = str(Path(*parts[-length:]))
                try:
                    candidate = _safe_resolve_under_roots(candidate_rel)
                    if candidate.exists() and candidate.is_file():
                        resolved = candidate
                        break
                except HTTPException:
                    continue
            if resolved is None:
                resolved = _find_file_by_name(p.name)
            if resolved is None:
                raise HTTPException(status_code=404, detail="File not found in any workspace root")
            p = resolved
    else:
        p = _safe_resolve_under_roots(raw)
        if not p.exists() or not p.is_file():
            fallback = _find_file_by_name(Path(raw).name)
            if fallback is None:
                raise HTTPException(status_code=404, detail="Not a file")
            p = fallback

    suffix = p.suffix.lower()
    size = p.stat().st_size
    limit = MAX_PDF_PREVIEW_BYTES if suffix == ".pdf" else MAX_PREVIEW_BYTES
    if size > limit:
        raise HTTPException(
            status_code=413,
            detail=f"File too large for preview ({size} bytes; max {limit})",
        )
    data = p.read_bytes()
    display_path = raw

    image_mimes = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".svg": "image/svg+xml",
    }
    if suffix == ".pdf":
        return {
            "kind": "pdf",
            "mime": "application/pdf",
            "content": base64.b64encode(data).decode("ascii"),
            "name": p.name,
            "path": display_path,
            "size": size,
        }
    if suffix in image_mimes:
        return {
            "kind": "image",
            "mime": image_mimes[suffix],
            "content": base64.b64encode(data).decode("ascii"),
            "name": p.name,
            "path": display_path,
            "size": size,
        }
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("latin-1")
    printable_ratio = sum(
        1 for c in text[:8000] if c.isprintable() or c in "\n\r\t"
    ) / max(min(len(text), 8000), 1)
    if printable_ratio < 0.85 and size > 256:
        return {
            "kind": "binary",
            "name": p.name,
            "path": display_path,
            "size": size,
            "message": "File looks binary; preview supports text, images, and PDF.",
        }
    return {
        "kind": "text",
        "content": text,
        "name": p.name,
        "path": display_path,
        "size": size,
    }


# ---------------------------------------------------------------------------
# Semantic search HTTP endpoints (direct access, independent of chat)
# ---------------------------------------------------------------------------


def _filter_by_score_gap(results: list[tuple[dict, float]], min_score: float = 0.60) -> list[tuple[dict, float]]:
    """Keep only results above *min_score* that sit above the largest score gap."""
    above = [(doc, s) for doc, s in results if s >= min_score]
    if len(above) <= 1:
        return above

    gaps = []
    for i in range(len(above) - 1):
        gaps.append((above[i][1] - above[i + 1][1], i))

    max_gap, gap_idx = max(gaps, key=lambda g: g[0])
    median_gap = sorted(g for g, _ in gaps)[len(gaps) // 2]

    if max_gap > median_gap * 2.5 and max_gap > 0.008:
        return above[: gap_idx + 1]

    return above


@app.get("/api/semantic/search")
def semantic_search(
    q: str = Query(..., min_length=1),
    k: int = Query(default=10, ge=1, le=20),
    min_score: float = Query(default=0.60, ge=0.0, le=1.0),
):
    try:
        from query_elements import similarity_search_with_score
    except ImportError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    try:
        results = similarity_search_with_score(q, k=k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    filtered = _filter_by_score_gap(results, min_score=min_score)

    hits = []
    for doc, score in filtered:
        hits.append(
            {
                "filename": doc.get("filename"),
                "filepath": doc.get("filepath"),
                "file_type": doc.get("file_type"),
                "score": round(score, 4),
            }
        )
    return {"query": q, "hits": hits}


class SemanticIndexBody(BaseModel):
    file_path: str = Field(..., min_length=1)
    description: str = ""


@app.post("/api/semantic/index")
def semantic_index(body: SemanticIndexBody):
    try:
        from add_element import ingest_file_to_db
    except ImportError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    p = Path(body.file_path).expanduser().resolve()
    if not p.is_file():
        raise HTTPException(status_code=400, detail=f"Not a file: {p}")
    if not _path_allowed(p):
        raise HTTPException(status_code=403, detail="File path outside allowed workspace roots")
    try:
        ingest_file_to_db(str(p), body.description or None)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return {"ok": True, "message": f"Indexed: {p.name}"}


_INDEX_EXTENSIONS = frozenset(
    {".pdf", ".png", ".jpg", ".jpeg", ".txt", ".md", ".webp", ".gif"}
)


class IndexDirectoryBody(BaseModel):
    rel_path: str = Field(..., min_length=1)


@app.post("/api/semantic/index_directory")
def semantic_index_directory(body: IndexDirectoryBody):
    try:
        from add_element import ingest_file_to_db
    except ImportError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    s = body.rel_path.strip()
    if os.path.isabs(s):
        root = Path(s).expanduser().resolve()
        if not root.is_dir():
            raise HTTPException(status_code=400, detail=f"Not a directory: {root}")
        if not _path_allowed(root):
            raise HTTPException(status_code=403, detail="Folder outside allowed roots.")
    else:
        root = _safe_resolve_under_roots(s)

    if not root.is_dir():
        raise HTTPException(status_code=400, detail="Not a directory")

    indexed = 0
    scanned_files = 0
    errors: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for name in filenames:
            if name.startswith("."):
                continue
            scanned_files += 1
            suf = Path(name).suffix.lower()
            if suf not in _INDEX_EXTENSIONS:
                continue
            fp = Path(dirpath) / name
            if not _path_allowed(fp):
                continue
            try:
                ingest_file_to_db(str(fp.resolve()), None)
                indexed += 1
            except Exception as e:
                errors.append(f"{fp.name}: {e}")

    return {
        "ok": True,
        "indexed": indexed,
        "scanned_files": scanned_files,
        "root_api": _rel_for_api(root),
        "resolved_absolute": str(root.resolve()),
        "errors": errors[:25],
    }


@app.get("/api/workspace/roots")
def workspace_roots():
    return {"roots": [str(r) for r in _content_roots()]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="127.0.0.1", port=8765, reload=False)
