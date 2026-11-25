"""PDF loading, chunking and indexing."""

from collections import defaultdict
from pathlib import Path
from typing import Iterable

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag import config

BATCH_SIZE = 64


def find_pdfs(paths: Iterable[str | Path]) -> list[Path]:
    """Expand files and directories into a sorted list of PDF paths."""
    pdfs: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            pdfs.extend(sorted(p.rglob("*.pdf")))
        elif p.is_file() and p.suffix.lower() == ".pdf":
            pdfs.append(p)
        else:
            raise FileNotFoundError(f"Not a PDF file or directory: {p}")
    return pdfs


def load_pdfs(pdf_paths: Iterable[Path]) -> list[Document]:
    """Load PDFs page by page, keeping source and 1-based page metadata."""
    pages: list[Document] = []
    for path in pdf_paths:
        for doc in PyPDFLoader(str(path)).load():
            doc.metadata = {
                "source": path.name,
                "path": str(path),
                "page": int(doc.metadata.get("page", 0)) + 1,
            }
            pages.append(doc)
    return pages


def split_documents(pages: list[Document]) -> list[Document]:
    """Split pages into 1,000-char chunks (200 overlap) and tag each with a chunk id."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        add_start_index=True,
    )
    chunks = [c for c in splitter.split_documents(pages) if c.page_content.strip()]

    counters: dict[str, int] = defaultdict(int)
    for chunk in chunks:
        source = chunk.metadata["source"]
        idx = counters[source]
        counters[source] += 1
        chunk.metadata["chunk"] = idx
        chunk.metadata["chunk_id"] = f"{source}:p{chunk.metadata['page']}:c{idx}"
    return chunks


def index_chunks(chunks: list[Document], store: Chroma) -> int:
    """Upsert chunks into Chroma using deterministic ids (re-ingesting is idempotent)."""
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        store.add_documents(batch, ids=[c.metadata["chunk_id"] for c in batch])
    return len(chunks)


def ingest(paths: Iterable[str | Path], store: Chroma, reset: bool = False) -> dict:
    pdfs = find_pdfs(paths)
    if not pdfs:
        raise FileNotFoundError("No PDF files found.")
    if reset:
        store.reset_collection()
    pages = load_pdfs(pdfs)
    chunks = split_documents(pages)
    index_chunks(chunks, store)
    return {"files": len(pdfs), "pages": len(pages), "chunks": len(chunks)}
