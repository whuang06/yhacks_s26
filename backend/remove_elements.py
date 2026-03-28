from pymongo import MongoClient
from bson.objectid import ObjectId
import os

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "yhacks"
COLLECTION_NAME = "files"

client = MongoClient(MONGO_URI)
collection = client[DB_NAME][COLLECTION_NAME]

def remove_by_filename(filename):
    """Removes a specific file by its name."""
    result = collection.delete_many({"filename": filename})
    print(f"Removed {result.deleted_count} instance(s) of '{filename}'.")

def remove_by_type(mime_type):
    """Removes all files of a certain type (e.g., 'image/jpeg')."""
    result = collection.delete_many({"file_type": mime_type})
    print(f"Removed {result.deleted_count} files of type '{mime_type}'.")

def remove_by_id(doc_id):
    """Removes a single document by its MongoDB ObjectID."""
    result = collection.delete_one({"_id": ObjectId(doc_id)})
    if result.deleted_count > 0:
        print(f"Successfully removed document ID: {doc_id}")
    else:
        print(f"No document found with ID: {doc_id}")

def reset_database():
    """Wipes the entire collection."""
    confirm = input("Are you sure you want to WIPE THE ENTIRE COLLECTION? (y/n): ")
    if confirm.lower() == 'y':
        result = collection.delete_many({})
        print(f"Database wiped! {result.deleted_count} documents deleted.")
    else:
        print("Reset cancelled.")

if __name__ == "__main__":
    remove_by_filename("person_portrait.jpeg")
    # reset_database()