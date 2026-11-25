# Plain-Vanilla RAG

A small RAG CLI for answering questions from PDF documents using Gemini, LangChain, and a local Chroma vector index.

## What it does

- Loads PDFs while preserving page metadata (`source`, `page` — 1-based).
- Splits text into **1,000-character chunks with 200-character overlap**.
- Embeds with **`gemini-embedding-001`** and retrieves the **top 4** chunks.
- Generates answers with **`gemini-2.5-flash`** using only the retrieved context.
- Returns **source, page, and chunk** citations.
- Uses a **fixed response** for unsupported questions:
  `I don't know based on the provided documents.`
- Includes an **LLM grounding evaluator** (LLM-as-judge) that checks whether every claim in an answer is supported by the retrieved chunks.

## Project layout

```
plain-vanilla-rag/
├── rag/
│   ├── __main__.py     # `python -m rag`
│   ├── cli.py          # ingest / ask / chat / eval / stats / reset
│   ├── config.py       # chunking, top-k, model names, fallback text
│   ├── ingest.py       # PDF loading, chunking, indexing
│   ├── vectorstore.py  # Gemini embeddings + Chroma
│   ├── qa.py           # retrieval, prompt, answer + citations
│   └── evaluator.py    # grounding evaluator
├── tests/              # offline tests (fake LLM/embeddings)
├── data/               # put your PDFs here
├── .env.example
└── requirements.txt
```

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then put your key in GOOGLE_API_KEY
```

Get an API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

## Usage

### 1. Index PDFs

```bash
python -m rag ingest data/                 # a directory (recursive)
python -m rag ingest report.pdf manual.pdf # or individual files
python -m rag ingest data/ --reset         # clear the index first
```

Chunk IDs are deterministic (`<file>:p<page>:c<chunk>`), so ingesting the same file again updates its chunks instead of duplicating them. The index is stored in `./chroma_db`.

### 2. Ask questions

```bash
python -m rag ask "What is the warranty period?"
```

```
Answer:
The warranty period is two years [1].

Sources:
  [1] manual.pdf, page 3, chunk 7
```

Options:

| Flag | Description |
|---|---|
| `--evaluate` | Also run the grounding evaluator on the answer |
| `--json` | Print the answer, citations, retrieved chunks (and evaluation) as JSON |
| `-k N` | Retrieve N chunks instead of 4 |

When the retrieved context does not answer the question:

```
Answer:
I don't know based on the provided documents.
```

### 3. Interactive chat

```bash
python -m rag chat --evaluate
```

### 4. Batch grounding evaluation

```bash
python -m rag eval data/sample_questions.txt --out eval_results.json
```

The questions file can be `.txt` (one per line, `#` comments allowed), `.json` (a list of strings or `{"question": ...}` objects), or `.jsonl` (`{"question": ...}` per line). For each question it prints whether the answer was grounded, was a fallback, or was **UNGROUNDED**, followed by a summary:

```json
{
  "questions": 3,
  "answered": 2,
  "fallbacks": 1,
  "grounded_rate": 1.0,
  "mean_score": 0.95
}
```

`grounded_rate` and `mean_score` are computed only over answered questions. Fallback answers make no claims, so they count as grounded.

### Other commands

```bash
python -m rag stats   # chunk counts per indexed file
python -m rag reset   # empty the index
```

## How it works

1. **Load:** `PyPDFLoader` reads each PDF into one document per page. Metadata is set to `source` (file name), `path`, and a 1-based `page`.
2. **Split:** `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)`. Each chunk gets a per-file `chunk` index and a `chunk_id`.
3. **Embed and store:** `GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")` writes to a persistent Chroma collection.
4. **Retrieve:** similarity search returns the top 4 chunks.
5. **Generate:** the chunks are numbered `[1]..[4]` with their source/page/chunk headers and passed to `gemini-2.5-flash` (temperature 0). The system prompt tells the model to use only that context, cite passage numbers, and reply with the exact fallback sentence when the context is not enough.
6. **Cite:** `[n]` markers in the answer are mapped back to source, page and chunk. If the model leaves out the markers, all retrieved chunks are listed.
7. **Evaluate (optional):** a separate judge call (`with_structured_output`) returns `grounded`, `score` (0–1), `unsupported_claims`, and `reasoning`. The judge looks only at the retrieved context, not at real-world truth.

## Configuration

Set these in `.env`:

| Variable | Default |
|---|---|
| `GOOGLE_API_KEY` | — (required) |
| `CHAT_MODEL` | `gemini-2.5-flash` |
| `EVAL_MODEL` | `gemini-2.5-flash` |
| `EMBEDDING_MODEL` | `models/gemini-embedding-001` |
| `CHROMA_DIR` | `./chroma_db` |
| `CHROMA_COLLECTION` | `pdf_rag` |

Chunk size, overlap, top-k and the fallback text are constants in `rag/config.py`.

> If you change `EMBEDDING_MODEL`, rebuild the index (`ingest --reset`). Vectors from different embedding models can't be compared.

## Tests

```bash
pytest -q
```

The tests use fake embeddings and a fake chat model, so they need no API key or network. They cover chunk sizes and metadata, idempotent re-ingest, citation mapping, the fallback path and the evaluator's fallback handling.

## Limitations

- Scanned or image-only PDFs have no text layer. Run OCR on them first.
- Retrieval is plain vector similarity, with no reranking, hybrid search or query rewriting. That is the "plain vanilla" part.
- The grounding evaluator is itself an LLM, so treat its scores as a signal, not ground truth.
