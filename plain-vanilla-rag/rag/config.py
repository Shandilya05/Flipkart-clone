"""Central configuration. Values can be overridden via environment / .env."""

import os

from dotenv import load_dotenv

load_dotenv()

# Chunking / retrieval (fixed by design)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4

# Models
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-2.5-flash")
EVAL_MODEL = os.getenv("EVAL_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")

# Vector store
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "pdf_rag")

# Fixed response used whenever the documents do not support an answer
FALLBACK_RESPONSE = "I don't know based on the provided documents."


def require_api_key() -> None:
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Copy .env.example to .env and add your key."
        )
