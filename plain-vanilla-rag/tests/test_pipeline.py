"""Offline tests: fake embeddings and fake chat models, real Chroma + splitter."""

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from rag import config
from rag.evaluator import evaluate_grounding
from rag.ingest import index_chunks, split_documents
from rag.qa import answer_question, extract_citations
from rag.vectorstore import get_vectorstore


@pytest.fixture
def pages():
    return [
        Document(page_content="Alpha " * 400, metadata={"source": "a.pdf", "page": 1}),
        Document(page_content="Bravo is a short page.", metadata={"source": "a.pdf", "page": 2}),
        Document(page_content="Charlie " * 50, metadata={"source": "b.pdf", "page": 1}),
    ]


@pytest.fixture
def store(tmp_path, pages):
    s = get_vectorstore(
        embeddings=DeterministicFakeEmbedding(size=32),
        persist_directory=str(tmp_path / "db"),
        collection_name="test",
    )
    index_chunks(split_documents(pages), s)
    return s


def test_split_sizes_and_metadata(pages):
    chunks = split_documents(pages)
    assert all(len(c.page_content) <= config.CHUNK_SIZE for c in chunks)
    a_chunks = [c for c in chunks if c.metadata["source"] == "a.pdf"]
    assert [c.metadata["chunk"] for c in a_chunks] == list(range(len(a_chunks)))
    assert a_chunks[-1].metadata["page"] == 2
    assert chunks[0].metadata["chunk_id"] == "a.pdf:p1:c0"
    # consecutive chunks on the long page overlap
    assert a_chunks[1].metadata["start_index"] < a_chunks[0].metadata["start_index"] + len(a_chunks[0].page_content)


def test_reingest_is_idempotent(store, pages):
    before = len(store.get()["ids"])
    index_chunks(split_documents(pages), store)
    assert len(store.get()["ids"]) == before


def test_answer_with_citations(store):
    llm = FakeListChatModel(responses=["Bravo is short [2]."])
    ans = answer_question("What is Bravo?", store, llm=llm)
    assert ans.supported
    assert len(ans.contexts) == config.TOP_K
    assert [c.ref for c in ans.citations] == [2]
    assert ans.citations[0].source == ans.contexts[1].metadata["source"]


def test_fallback_response(store):
    llm = FakeListChatModel(responses=[config.FALLBACK_RESPONSE])
    ans = answer_question("Capital of Mars?", store, llm=llm)
    assert not ans.supported
    assert ans.answer == config.FALLBACK_RESPONSE
    assert ans.citations == []


def test_uncited_answer_cites_all_retrieved():
    docs = [Document(page_content="x", metadata={"source": "s", "page": 1, "chunk": i}) for i in range(3)]
    assert [c.ref for c in extract_citations("No markers here", docs)] == [1, 2, 3]
    assert [c.ref for c in extract_citations("See [3] and [9] and [3]", docs)] == [3]


def test_evaluator_skips_fallback():
    v = evaluate_grounding("q", config.FALLBACK_RESPONSE, [], judge=None)
    assert v.grounded and v.score == 1.0
