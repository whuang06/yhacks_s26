"""
Update stored fields on existing documents without re-ingesting.

Use $set for metadata / paths: no delete+reinsert needed.

Re-embed (re-run get_multimodal_embedding + $set embedding) only if the *file bytes*
changed and you want search quality to match the new content.
"""
import os
from typing import Any

from bson.objectid import ObjectId
from pymongo import MongoClient
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "yhacks"
COLLECTION_NAME = "files"

client = MongoClient(MONGO_URI)
collection = client[DB_NAME][COLLECTION_NAME]


def update_entries(
    filter_query: dict[str, Any],
    set_fields: dict[str, Any],
    *,
    many: bool = False,
) -> tuple[int, int]:
    """
    $set top-level or dotted fields (e.g. {"filepath": "...", "metadata.file_size": 1024}).

    Returns (matched_count, modified_count).
    """
    if not set_fields:
        raise ValueError("set_fields must not be empty")

    if many:
        result = collection.update_many(filter_query, {"$set": set_fields})
    else:
        result = collection.update_one(filter_query, {"$set": set_fields})
    return result.matched_count, result.modified_count


def update_by_id(
    doc_id: str,
    set_fields: dict[str, Any],
) -> tuple[int, int]:
    """Patch a single document by _id."""
    return update_entries({"_id": ObjectId(doc_id)}, set_fields, many=False)


def update_by_filename(
    filename: str,
    set_fields: dict[str, Any],
    *,
    many: bool = False,
) -> tuple[int, int]:
    """Patch by exact filename (basename as stored). Use many=True if duplicates exist."""
    return update_entries({"filename": filename}, set_fields, many=many)


def _filepath_set_fields(new_filepath: str, *, abs_path: bool) -> dict[str, Any]:
    path = os.path.abspath(new_filepath) if abs_path else new_filepath
    fields: dict[str, Any] = {"filepath": path}
    if abs_path:
        fields["filename"] = os.path.basename(path)
    return fields


def update_filepath_by_id(
    doc_id: str,
    new_filepath: str,
    *,
    abs_path: bool = True,
) -> tuple[int, int]:
    """Look up by `_id` (hex string). Sets `filepath` and, if abs_path, `filename` from basename."""
    return update_by_id(doc_id, _filepath_set_fields(new_filepath, abs_path=abs_path))


def update_filepath(
    filename: str,
    new_filepath: str,
    *,
    abs_path: bool = True,
    many: bool = False,
) -> tuple[int, int]:
    """
    Look up by stored basename `filename` (Mongo field `filename`, not full path).
    Sets `filepath` to `new_filepath`; if abs_path, also sets `filename` to basename(new_filepath).
    """
    return update_by_filename(
        filename, _filepath_set_fields(new_filepath, abs_path=abs_path), many=many
    )


def get_one_by_filename(filename: str) -> dict[str, Any] | None:
    return collection.find_one({"filename": filename})


if __name__ == "__main__":

    update_filepath_by_id(
        "69c8280d2c2cc24b44e5d86d",
        "/absolute/path/to/person_portrait.jpeg",
    )
    
    update_filepath_by_id(
        "69c8280d2c2cc24b44e5d86d",
        "/Users/william/yhacks_s26/exact_names_final/5xm98t580u/person_portrait.jpeg",
    )