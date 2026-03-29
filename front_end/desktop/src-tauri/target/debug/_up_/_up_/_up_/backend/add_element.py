import os
import magic
from google import genai
from google.genai import types
from pymongo import MongoClient
from input_to_embedding import get_multimodal_embedding

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "yhacks"
COLLECTION_NAME = "files"

client = genai.Client(api_key=GEMINI_API_KEY)
db_client = MongoClient(MONGO_URI)
collection = db_client[DB_NAME][COLLECTION_NAME]

def ingest_file_to_db(file_path, description=None):
    """
    Generates an embedding for a file and stores it in MongoDB
    with associated metadata.

    Returns the inserted document's ``_id``, or ``None`` if the file is missing
    or embedding failed.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None

    file_name = os.path.basename(file_path)
    mime_type = magic.from_file(file_path, mime=True)

    embedding = get_multimodal_embedding(file_path, description)
    if embedding is None:
        print("Error: No embedding produced; not inserting.")
        return None

    # Prepare the Document
    document = {
        "filename": file_name,
        "filepath": os.path.abspath(file_path),
        "file_type": mime_type,
        "embedding": embedding,
        "metadata": {
            "file_size": os.path.getsize(file_path),
            "description_provided": bool(description)
        }
    }
    try:
        result = collection.insert_one(document)
        print(f"Successfully indexed {file_name} (ID: {result.inserted_id})")
        return result.inserted_id
    except Exception as e:
        print(f"Failed to insert document: {e}")
        raise e


if __name__ == "__main__":
    target_file = "/Users/william/yhacks_s26/test_directory/3vhwp7pn/cat.jpeg"
    ingest_file_to_db(target_file)