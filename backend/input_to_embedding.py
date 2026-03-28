import os
import magic
from google import genai
from google.genai import types
from google.api_core import exceptions

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def get_multimodal_embedding(file_path, description=None, task_type="RETRIEVAL_DOCUMENT"):
    """
    Sends raw file bytes (PDF, Image, Text) directly to Gemini 2 
    without manual extraction.
    """
    mime_type = magic.from_file(file_path, mime=True)
    content_parts = []
    
    if description:
        content_parts.append(description)

    try:
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        
        file_part = types.Part.from_bytes(
            data=file_bytes, 
            mime_type=mime_type
        )
        content_parts.append(file_part)
        print(f"Prepared {mime_type} for direct embedding: {os.path.basename(file_path)}")

    except Exception as e:
        print(f"Error reading file: {e}")
        return None

    try:
        res = client.models.embed_content(
            model="gemini-embedding-2-preview",
            contents=content_parts,
            config={
                'task_type': task_type,
                'output_dimensionality': 768
            }
        )
        return res.embeddings[0].values
    except exceptions.InvalidArgument as e:
        print(f"Validation Error: Ensure the model supports {mime_type}. Error: {e}")
        return None
    except Exception as e:
        print(f"General Error: {e}")
        return None