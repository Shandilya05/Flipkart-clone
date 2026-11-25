"""Embedding model and Chroma vector store factories."""

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from rag import config


def get_embeddings() -> Embeddings:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    config.require_api_key()
    return GoogleGenerativeAIEmbeddings(model=config.EMBEDDING_MODEL)


def get_vectorstore(
    embeddings: Embeddings | None = None,
    persist_directory: str | None = None,
    collection_name: str | None = None,
) -> Chroma:
    return Chroma(
        collection_name=collection_name or config.CHROMA_COLLECTION,
        embedding_function=embeddings or get_embeddings(),
        persist_directory=persist_directory or config.CHROMA_DIR,
    )
