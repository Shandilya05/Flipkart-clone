"""Retrieval + grounded answer generation with citations."""

import re
from dataclasses import asdict, dataclass, field

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore

from rag import config

SYSTEM_PROMPT = f"""You are a question-answering assistant for a set of PDF documents.

Rules:
- Answer ONLY with information found in the numbered context passages below.
- Do not use prior knowledge, and do not guess or speculate.
- Cite every claim with the passage number(s) in square brackets, e.g. [1] or [2][3].
- Be concise and factual.
- If the context does not contain enough information to answer, reply with exactly:
{config.FALLBACK_RESPONSE}
"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)

_CITATION_RE = re.compile(r"\[(\d+)\]")


@dataclass
class Citation:
    ref: int
    source: str
    page: int
    chunk: int
    chunk_id: str

    def __str__(self) -> str:
        return f"[{self.ref}] {self.source}, page {self.page}, chunk {self.chunk}"


@dataclass
class Answer:
    question: str
    answer: str
    supported: bool
    citations: list[Citation] = field(default_factory=list)
    contexts: list[Document] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "supported": self.supported,
            "citations": [asdict(c) for c in self.citations],
            "contexts": [
                {"text": d.page_content, **d.metadata} for d in self.contexts
            ],
        }


def get_llm(model: str | None = None) -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI

    config.require_api_key()
    return ChatGoogleGenerativeAI(model=model or config.CHAT_MODEL, temperature=0)


def format_context(docs: list[Document]) -> str:
    blocks = []
    for i, d in enumerate(docs, start=1):
        m = d.metadata
        header = f"[{i}] (source: {m.get('source')}, page: {m.get('page')}, chunk: {m.get('chunk')})"
        blocks.append(f"{header}\n{d.page_content}")
    return "\n\n".join(blocks)


def to_citation(ref: int, doc: Document) -> Citation:
    m = doc.metadata
    return Citation(
        ref=ref,
        source=str(m.get("source", "unknown")),
        page=int(m.get("page", 0)),
        chunk=int(m.get("chunk", 0)),
        chunk_id=str(m.get("chunk_id", "")),
    )


def is_fallback(text: str) -> bool:
    normalized = text.strip().strip('"').strip()
    return normalized == "" or config.FALLBACK_RESPONSE.lower() in normalized.lower()


def extract_citations(text: str, docs: list[Document]) -> list[Citation]:
    """Return citations for passage numbers referenced in the answer (in order)."""
    refs: list[int] = []
    for match in _CITATION_RE.findall(text):
        n = int(match)
        if 1 <= n <= len(docs) and n not in refs:
            refs.append(n)
    if not refs:  # model forgot to cite: fall back to all retrieved passages
        refs = list(range(1, len(docs) + 1))
    return [to_citation(n, docs[n - 1]) for n in refs]


def retrieve(question: str, store: VectorStore, k: int = config.TOP_K) -> list[Document]:
    return store.similarity_search(question, k=k)


def answer_question(
    question: str,
    store: VectorStore,
    llm: BaseChatModel | None = None,
    k: int = config.TOP_K,
) -> Answer:
    docs = retrieve(question, store, k=k)
    if not docs:
        return Answer(question, config.FALLBACK_RESPONSE, supported=False)

    chain = PROMPT | (llm or get_llm()) | StrOutputParser()
    text = chain.invoke({"context": format_context(docs), "question": question}).strip()

    if is_fallback(text):
        return Answer(question, config.FALLBACK_RESPONSE, supported=False, contexts=docs)

    return Answer(
        question=question,
        answer=text,
        supported=True,
        citations=extract_citations(text, docs),
        contexts=docs,
    )
