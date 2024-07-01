import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
API_KEY_PRIMARY = os.getenv("API_KEY_PRIMARY_VAL")

CHUNKING_API_URL = os.getenv("CHUNKING_API_URL")
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL")

COLLECTION_DESCRIPTION = os.getenv("COLLECTION_DESCRIPTION")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
