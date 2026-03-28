# YHacks 2026 — Copper Golem

Multimodal file embeddings (PDF, images, text) with **Google Gemini**, stored in **MongoDB Atlas**, and queried with **Atlas Vector Search** (`$vectorSearch`).

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

**Database:** `yhacks`  
**Collection:** `files`  
**Vector index name:** `vector_index` (must match `VECTOR_INDEX_NAME` in `query_elements.py`)  
**Vector field:** `embedding` (array of 768 floats)

Stored documents include at least: `filename`, `filepath`, `file_type`, `embedding`, and `metadata` (see `add_element.py`).

---

## Prerequisites

- **Python 3.11+** (3.13 works)
- **MongoDB Atlas** cluster with **Vector Search** enabled — `$vectorSearch` does not run against a default local `mongod`
- **Gemini API key** with access to embedding models
- **libmagic** (for `python-magic` / `magic.from_file`):
  - macOS: `brew install libmagic`

---

## Setup

```bash
cd yhacks_s26
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install pymongo google-genai google-api-core python-magic
```

Install from the **`backend`** directory when running scripts there (or use `python -m` with `PYTHONPATH` set to `backend` if you prefer).

---

## Environment variables

| Variable | Required | Notes |
|----------|----------|--------|
| `MONGO_URI` | Yes | Atlas SRV connection string. If unset, PyMongo defaults to **`localhost:27017`**, which has **no** vector index — searches return `[]` with no obvious error. |
| `GEMINI_API_KEY` | Yes | Used by `google.genai` for embeddings. |

Example:

```bash
export MONGO_URI='mongodb+srv://USER:PASS@cluster.mongodb.net/?appName=...'
export GEMINI_API_KEY='your-key'
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

---

## Backend scripts (quick reference)

| Script | Purpose |
|--------|---------|
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
