"""
LangGraph + LangChain agent: tools (MongoDB vector + embeddings), graph, CLI.

    cd backend
    export GEMINI_API_KEY=...
    export MONGO_URI=...
    pip install langchain-core langchain-google-genai langgraph
    python agent.py

Extend: add ``@tool`` functions below, append them to ``AGENT_TOOLS``, and describe
them clearly in the docstrings (the model uses those for routing).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure ``backend/`` is on path when running ``python agent.py`` from any cwd.
_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from bson.objectid import ObjectId
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

# --- Backend modules (sibling imports; cwd should be backend/) ---
from add_element import ingest_file_to_db
from query_elements import similarity_search_with_score
from update_element import update_filepath_by_id

import remove_elements as remove_mod


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
        lines.append(
            f"{i}. score={score:.4f} | {doc.get('filename', '?')} | {doc.get('file_type', '?')}\n"
            f"   path: {doc.get('filepath', '?')}"
        )
    return "\n".join(lines)


@tool
def index_file_to_vector_database(absolute_file_path: str, description: str = "") -> str:
    """Embed a file (PDF, image, text, …) with Gemini and insert into MongoDB ``yhacks.files``.

    ``absolute_file_path`` must exist. Optional ``description`` adds context for the embedding.
    """
    path = Path(absolute_file_path).expanduser().resolve()
    if not path.is_file():
        return f"Not a file or missing: {path}"
    try:
        ingest_file_to_db(str(path), description or None)
    except Exception as e:
        return f"Ingest failed: {e}"
    return f"Indexed: {path.name}"


@tool
def remove_vector_entry_by_filename(filename: str) -> str:
    """Remove all MongoDB documents whose stored ``filename`` (basename) matches."""
    result = remove_mod.collection.delete_many({"filename": filename})
    return f"Deleted {result.deleted_count} document(s) with filename={filename!r}."


@tool
def remove_vector_entry_by_id(mongo_object_id: str) -> str:
    """Remove one document by MongoDB ``_id`` (24-character hex string)."""
    try:
        oid = ObjectId(mongo_object_id)
    except Exception:
        return f"Invalid ObjectId: {mongo_object_id!r}"
    result = remove_mod.collection.delete_one({"_id": oid})
    if result.deleted_count:
        return f"Deleted document _id={mongo_object_id}."
    return f"No document with _id={mongo_object_id!r}."


@tool
def update_stored_filepath_by_document_id(
    mongo_object_id: str,
    new_absolute_filepath: str,
) -> str:
    """Update ``filepath`` / ``filename`` for one row by ``_id`` (e.g. after moving a file on disk)."""
    try:
        matched, modified = update_filepath_by_id(mongo_object_id, new_absolute_filepath)
    except Exception as e:
        return f"Update failed: {e}"
    return f"matched={matched}, modified={modified}"


AGENT_TOOLS = [
    semantic_file_search,
    index_file_to_vector_database,
    remove_vector_entry_by_filename,
    remove_vector_entry_by_id,
    update_stored_filepath_by_document_id,
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
    messages: list = []
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
