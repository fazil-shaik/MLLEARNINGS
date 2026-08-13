import os

from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "simple_rag"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200