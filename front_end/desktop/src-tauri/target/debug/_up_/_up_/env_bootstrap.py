"""Load project .env files and strip blank secrets so real keys are not masked."""

from __future__ import annotations

import os
from pathlib import Path

_STRIP_IF_BLANK = (
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "MONGO_URI",
    "MONGODB_URI",
)


def load_project_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    root = Path(__file__).resolve().parent
    repo_root = root.parent
    for env_path in (
        root / ".env",
        root / "desktop" / ".env",
        repo_root / ".env",
    ):
        if env_path.is_file():
            load_dotenv(env_path, override=False)

    for key in _STRIP_IF_BLANK:
        val = os.environ.get(key)
        if val is not None and not str(val).strip():
            del os.environ[key]

    # YHACK_ROOT is a friendly alias for YHACKS_FS_ROOT (agent + preview roots).
    yroot = (os.environ.get("YHACK_ROOT") or "").strip()
    if yroot:
        p = Path(yroot).expanduser()
        if p.is_absolute():
            fs = str(p.resolve())
        else:
            resolved = None
            here = Path(__file__).resolve().parent
            for base in (Path.home() / "Downloads", Path.cwd(), here, here.parent):
                cand = (base / yroot).resolve()
                if cand.is_dir():
                    resolved = str(cand)
                    break
            fs = resolved or str((Path.cwd() / yroot).resolve())
        os.environ["YHACKS_FS_ROOT"] = fs
        if not (os.environ.get("COPPERGOLEM_EXTRA_ROOTS") or "").strip():
            os.environ["COPPERGOLEM_EXTRA_ROOTS"] = fs
