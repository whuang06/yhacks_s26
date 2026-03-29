# Sift

Multimodal file embeddings (PDF, images, text) with **Google Gemini**, stored in **MongoDB Atlas**, queried with **Atlas Vector Search** (`$vectorSearch`), plus a **LangGraph + LangChain** chat agent that can search the index, run filesystem plans, trash files, and **ask questions about local file contents** (multimodal Gemini).

---

## Architecture

| Piece | Role |
|--------|------|
| **`input_to_embedding.py`** | Calls `gemini-embedding-2-preview`: `RETRIEVAL_DOCUMENT` for files, `RETRIEVAL_QUERY` for search text. Vectors are **768** dimensions. |
| **`add_element.py`** | Ingests one file: MIME via `python-magic`, embedding via Gemini, `insert_one` into Atlas. |
| **`batch_process.py`** | Walks a directory and calls `ingest_file_to_db` for each allowed extension. |
| **`create_vector_index.py`** | **One-time** (per index): creates the Atlas **Vector Search** index on the `embedding` field. Not part of the aggregation pipeline. |
| **`query_elements.py`** | Builds a pipeline (`$vectorSearch` → `$project`) and runs **`collection.aggregate(pipeline)`** — this is where the pipeline executes. |
| **`remove_elements.py`** | Deletes documents by filename, MIME type, or `_id`. |
| **`update_element.py`** | **`$set`** updates (e.g. `filepath`, `filename`, nested `metadata.*`) without re-ingesting. |
| **`mongo_test_connect.py`** | Ping Atlas with `MONGO_URI` to verify connectivity. |
| **`agent.py`** | **LangGraph** agent: `ChatGoogleGenerativeAI` + tools (`semantic_file_search`, `ask_about_files`, `trash_file`, `preview_plan`, `execute_plan`, `undo_last_action`). REPL: `python agent.py`. |

**Database:** `yhacks`  
**Collection:** `files`  
**Vector index name:** `vector_index` (must match `VECTOR_INDEX_NAME` in `query_elements.py`)  
**Vector field:** `embedding` (array of 768 floats)

Stored documents include at least: `filename`, `filepath`, `file_type`, `embedding`, and `metadata` (see `add_element.py`).

---

## LangGraph agent (`backend/agent.py`)

Interactive CLI: from `backend/`, with venv activated:

```bash
cd backend
source ../venv/bin/activate   # adjust path if needed
python agent.py
```

Empty line exits. The agent uses **`GEMINI_API_KEY`** and optionally **`MONGO_URI`** for vector search.

### Environment (agent)

| Variable | Required | Notes |
|----------|----------|--------|
| `GEMINI_API_KEY` | Yes | Gemini chat + embeddings. |
| `MONGO_URI` | For search tools | Needed for `semantic_file_search`; omit only if you only use non-index tools. |
| `YHACKS_FS_ROOT` | No | All filesystem paths in tools/plans must stay under this directory. Default: **current working directory** when the process starts. Set it to your project root for predictable behavior. |
| `AGENT_MODEL` | No | Gemini model id for the agent and `ask_about_files`. Default: **`gemini-2.5-flash`**. |

### Tools (registered on the model)

| Tool | Purpose |
|------|---------|
| **`semantic_file_search`** | Natural-language search over **indexed** files (`similarity_search_with_score`). Args: `query`, optional `k` (1–20). |
| **`ask_about_files`** | **Direct Q&A** on one or more **local files** via **`ChatGoogleGenerativeAI`** + multimodal `HumanMessage` (inline bytes + MIME). Does **not** require MongoDB. Args: `question`, `file_paths`. |
| **`trash_file`** | Move a file to **system Trash** (`send2trash`), remove matching vector row; may record undo on macOS/Linux (see undo). |
| **`preview_plan`** | Show what a JSON plan would do (no side effects). |
| **`execute_plan`** | Run a JSON plan; always previews first; optional `dry_run=True`. Records one **undo batch** per successful run. |
| **`undo_last_action`** | Reverses the last batch from `execute_plan` or `trash_file` (LIFO). |

### `ask_about_files` — paths and limits

- Paths are resolved under **`YHACKS_FS_ROOT`** (or cwd).
- **`file_paths`**: either a **JSON array** of relative paths, e.g. `["notes/a.pdf","img/b.png"]`, or a **single** relative path as plain text.
- **Up to 10 files** per call; **up to 100 MiB per file** (guard in code; Gemini still enforces its own request/token limits).
- MIME detection: `python-magic` when installed, else `mimetypes` fallback.

### Plan JSON (`execute_plan` / `preview_plan`)

`plan_json` is a JSON **array** of steps. Common actions:

| Action | Fields | Notes |
|--------|--------|--------|
| `create_folder` | `path` | Creates directory under project root. |
| `move_file` | `from`, `to`, optional `mongo_id` | Moves file; updates Mongo by `_id` or by `filepath`. |
| `remove_file` | `path` and/or `mongo_id` | Uses **send2trash** (system Trash) + removes DB rows; on **macOS/Linux** undo may restore from Trash + re-index. |
| `add_file` | `path`, optional `description` | Runs `ingest_file_to_db`. Undo removes only the new Mongo row. |
| `remove_folder` | `path` | Trash + delete DB rows under path; **not** recorded for agent undo. |

`execute_plan` runs the same preview as `preview_plan` before executing. Use `dry_run=True` to preview only.

### Python dependencies (agent)

In addition to the core stack, the agent needs:

`langchain-core`, `langchain-google-genai`, `langgraph`, `send2trash` (for trash/removals).

Example:

```bash
pip install langchain-core langchain-google-genai langgraph send2trash pymongo google-genai google-api-core python-magic
```

---

## Prerequisites

- **Python 3.11+** (3.13 works)
- **MongoDB Atlas** cluster with **Vector Search** enabled — `$vectorSearch` does not run against a default local `mongod`
- **Gemini API key** with access to embedding and chat models
- **libmagic** (for `python-magic` / `magic.from_file`):
  - macOS: `brew install libmagic`

---

## Setup

```bash
cd yhacks_s26
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install pymongo google-genai google-api-core python-magic
pip install langchain-core langchain-google-genai langgraph send2trash
```

Install from the **`backend`** directory when running scripts there (or use `python -m` with `PYTHONPATH` set to `backend` if you prefer).

---

## Environment variables

| Variable | Required | Notes |
|----------|----------|--------|
| `MONGO_URI` | Yes (for index/search) | Atlas SRV connection string. If unset, PyMongo defaults to **`localhost:27017`**, which has **no** vector index — searches return `[]` with no obvious error. |
| `GEMINI_API_KEY` | Yes | Used by `google.genai` for embeddings and by the LangChain agent. |
| `YHACKS_FS_ROOT` | No | Agent / plan filesystem root (see above). |
| `AGENT_MODEL` | No | Default `gemini-2.5-flash`. |

Example:

```bash
export MONGO_URI='mongodb+srv://USER:PASS@cluster.mongodb.net/?appName=...'
export GEMINI_API_KEY='your-key'
export YHACKS_FS_ROOT='/absolute/path/to/your/project'   # optional
```

---

## Workflow (first time)

1. **Create the vector search index** (once per collection / definition):

   ```bash
   cd backend
   source ../venv/bin/activate
   python create_vector_index.py
   ```

2. In **Atlas UI**: open the cluster → **Database** → `yhacks` → `files` → **Search Indexes** — wait until the index named **`vector_index`** is **READY** (can take a few minutes).

3. **Ingest files**:

   - Single file: run `add_element.py` (adjust the path in `if __name__ == "__main__"`), or import `ingest_file_to_db`.
   - Many files: set `TARGET_DIRECTORY` in `batch_process.py` to your folder, then:

     ```bash
     python batch_process.py
     ```

4. **Query**:

   ```bash
   python query_elements.py
   ```

   Or call `similarity_search_with_score(query="...", k=3)` from code.

5. **Chat agent** (optional):

   ```bash
   cd backend
   python agent.py
   ```

**Order note:** You can ingest before or after creating the index; the index must be **READY** before `aggregate()` with `$vectorSearch` returns meaningful hits.

### Updating paths or metadata

You do **not** need to remove and re-add a document just to change **`filepath`**, **`filename`**, or **`metadata`** — use **`update_element.py`** (MongoDB **`$set`**) so the same `embedding` stays attached.

Re-run **`get_multimodal_embedding`** and **`$set` the `embedding` field** only if the **file contents** changed and you want similarity search to reflect the new bytes.

---

## How the search pipeline runs

Index creation (`create_search_index` / Atlas UI) only **registers** the vector index.  
The **pipeline runs** when you call:

```python
collection.aggregate(pipeline)
```

That happens in `query_elements.py` inside `similarity_search_with_score`: the first stage is `$vectorSearch` (uses index name **`vector_index`**, path **`embedding`**, and the live query vector from Gemini), then `$project` adds `vectorSearchScore` as `score`.

---

## Troubleshooting

| Symptom | Likely cause |
|---------|----------------|
| Always `[]` | No vector index on `yhacks.files`, or index not **READY**. |
| Always `[]` | `MONGO_URI` not set → connecting to localhost instead of Atlas. |
| Always `[]` | Index **name** in Atlas does not match `VECTOR_INDEX_NAME` (`vector_index`). |
| Errors or bad results | Index **numDimensions** ≠ **768** or wrong **`path`** (must be `embedding`). |
| `import magic` / libmagic errors | Install **libmagic** (`brew install libmagic` on macOS). |
| Agent trash / remove fails | `pip install send2trash`. |
| `ask_about_files` fails on huge PDFs | Within-code cap is 100 MiB/file; Gemini may still reject or truncate very large inputs—use smaller files, File API, or chunking for production. |

---

## Backend scripts (quick reference)

| Script | Purpose |
|--------|---------|
| `agent.py` | LangGraph REPL: tools for search, Q&A on files, plans, trash, undo. |
| `create_vector_index.py` | Create `vector_index` on `embedding` (768-dim, cosine) + filter field `file_type`. |
| `batch_process.py` | Batch ingest; edit `TARGET_DIRECTORY` and optional `ALLOWED_EXTENSIONS`. |
| `add_element.py` | Single-file ingest example in `__main__`. |
| `query_elements.py` | Semantic search CLI / `similarity_search_with_score`. |
| `remove_elements.py` | Delete by filename, MIME type, or id. |
| `update_element.py` | Patch in place: `update_filepath_by_id`, `update_filepath` (by basename), `update_entries`, … CLI: `python update_element.py <ObjectId> <new_filepath>`. |
| `mongo_test_connect.py` | Connection smoke test. |
| `input_embedding_sample_test.py` | Quick test of `get_multimodal_embedding`. |

---

## Allowed batch extensions

Default in `batch_process.py`: `.pdf`, `.png`, `.jpg`, `.jpeg`, `.txt`, `.md`
