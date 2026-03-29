import os
from add_element import ingest_file_to_db

TARGET_DIRECTORY = "/Users/william/yhacks_s26/test_directory"
ALLOWED_EXTENSIONS = {'.pdf', '.png', '.jpg', '.jpeg', '.txt', '.md'}

def process_directory(directory_path):
    """
    Recursively finds all files in a directory and adds them 
    to the vector database.
    """
    if not os.path.exists(directory_path):
        print(f"Error: Directory '{directory_path}' does not exist.")
        return

    print(f"Starting batch ingestion for: {os.path.abspath(directory_path)}")
    
    file_count = 0
    success_count = 0

    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.startswith('.'):
                continue
                
            ext = os.path.splitext(file)[1].lower()
            if ext not in ALLOWED_EXTENSIONS:
                print(f"Skipping unsupported file: {file}")
                continue

            file_path = os.path.join(root, file)
            file_count += 1
            
            print(f"({file_count}) Processing: {file}...")
            
            try:
                ingest_file_to_db(file_path)
                success_count += 1
            except Exception as e:
                print(f"Failed to ingest {file}: {e}")

    print(f"Ingestion complete")
    print(f"Total files found: {file_count}")
    print(f"Successfully indexed: {success_count}")

if __name__ == "__main__":
    process_directory(TARGET_DIRECTORY)