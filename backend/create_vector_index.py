"""
One-shot: create Atlas Vector Search index via PyMongo.
Run once; building the index can take a minute in Atlas UI.

Requires: MONGO_URI, Atlas cluster with Vector Search enabled.
Dimensions must match input_to_embedding.py (Gemini output_dimensionality).
"""
import os

from pymongo import MongoClient
from pymongo.operations import SearchIndexModel

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "yhacks"
COLLECTION_NAME = "files"
INDEX_NAME = "vector_index"
EMBEDDING_DIMS = 768

search_index_model = SearchIndexModel(
    definition={
        "fields": [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": EMBEDDING_DIMS,
                "similarity": "cosine",
            },
            {
                "type": "filter",
                "path": "file_type",
            },
        ]
    },
    name=INDEX_NAME,
    type="vectorSearch",
)

if __name__ == "__main__":
    if not MONGO_URI:
        raise SystemExit("Set MONGO_URI")

    collection = MongoClient(MONGO_URI)[DB_NAME][COLLECTION_NAME]
    collection.create_search_index(model=search_index_model)
    print(
        f"Created search index {INDEX_NAME!r} on {DB_NAME}.{COLLECTION_NAME} "
        f"({EMBEDDING_DIMS} dims). Wait until Atlas shows index READY before querying."
    )
