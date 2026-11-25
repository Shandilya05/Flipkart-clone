"""Command-line interface: ingest, ask, chat, eval, stats, reset."""

import argparse
import json
import sys
from pathlib import Path

from rag import config


def _store():
    from rag.vectorstore import get_vectorstore

    return get_vectorstore()


def _print_answer(ans, verdict=None, as_json=False) -> None:
    if as_json:
        out = ans.to_dict()
        if verdict is not None:
            out["evaluation"] = verdict.model_dump()
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return

    print(f"\nAnswer:\n{ans.answer}\n")
    if ans.citations:
        print("Sources:")
        for c in ans.citations:
            print(f"  {c}")
    if verdict is not None:
        status = "GROUNDED" if verdict.grounded else "NOT GROUNDED"
        print(f"\nGrounding: {status} (score {verdict.score:.2f})")
        print(f"  {verdict.reasoning}")
        for claim in verdict.unsupported_claims:
            print(f"  - unsupported: {claim}")
    print()


def cmd_ingest(args) -> int:
    from rag.ingest import ingest

    stats = ingest(args.paths, _store(), reset=args.reset)
    print(f"Indexed {stats['chunks']} chunks from {stats['pages']} pages in {stats['files']} file(s).")
    print(f"Index location: {Path(config.CHROMA_DIR).resolve()}")
    return 0


def cmd_ask(args) -> int:
    from rag.qa import answer_question

    ans = answer_question(args.question, _store(), k=args.k)
    verdict = None
    if args.evaluate:
        from rag.evaluator import evaluate_answer

        verdict = evaluate_answer(ans)
    _print_answer(ans, verdict, as_json=args.json)
    return 0


def cmd_chat(args) -> int:
    from rag.qa import answer_question

    store = _store()
    evaluate = None
    if args.evaluate:
        from rag.evaluator import evaluate_answer as evaluate
    print("Ask questions about your PDFs. Type 'exit' or press Ctrl-D to quit.")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if q.lower() in {"exit", "quit"}:
            return 0
        if not q:
            continue
        ans = answer_question(q, store, k=args.k)
        _print_answer(ans, evaluate(ans) if evaluate else None)


def _read_questions(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line)["question"] for line in text.splitlines() if line.strip()]
    if path.suffix == ".json":
        data = json.loads(text)
        return [q if isinstance(q, str) else q["question"] for q in data]
    return [l.strip() for l in text.splitlines() if l.strip() and not l.startswith("#")]


def cmd_eval(args) -> int:
    from rag.evaluator import evaluate_answer
    from rag.qa import answer_question

    questions = _read_questions(Path(args.questions))
    store = _store()
    results = []
    for i, q in enumerate(questions, start=1):
        ans = answer_question(q, store, k=args.k)
        verdict = evaluate_answer(ans)
        results.append({**ans.to_dict(), "evaluation": verdict.model_dump()})
        flag = "fallback" if not ans.supported else ("ok" if verdict.grounded else "UNGROUNDED")
        print(f"[{i}/{len(questions)}] {flag:<10} score={verdict.score:.2f}  {q}")

    answered = [r for r in results if r["supported"]]
    summary = {
        "questions": len(results),
        "answered": len(answered),
        "fallbacks": len(results) - len(answered),
        "grounded_rate": (sum(r["evaluation"]["grounded"] for r in answered) / len(answered)) if answered else None,
        "mean_score": (sum(r["evaluation"]["score"] for r in answered) / len(answered)) if answered else None,
    }
    print("\nSummary:", json.dumps(summary, indent=2))
    if args.out:
        for r in results:
            if not args.include_contexts:
                r.pop("contexts", None)
        Path(args.out).write_text(
            json.dumps({"summary": summary, "results": results}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Results written to {args.out}")
    return 0


def cmd_stats(args) -> int:
    store = _store()
    data = store.get(include=["metadatas"])
    sources: dict[str, int] = {}
    for m in data["metadatas"]:
        sources[m.get("source", "unknown")] = sources.get(m.get("source", "unknown"), 0) + 1
    print(f"Collection '{config.CHROMA_COLLECTION}' at {Path(config.CHROMA_DIR).resolve()}")
    print(f"Total chunks: {len(data['ids'])}")
    for s, n in sorted(sources.items()):
        print(f"  {s}: {n} chunks")
    return 0


def cmd_reset(args) -> int:
    _store().reset_collection()
    print(f"Collection '{config.CHROMA_COLLECTION}' cleared.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rag", description="Plain-Vanilla RAG over PDF documents.")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("ingest", help="Load, chunk and index PDFs.")
    s.add_argument("paths", nargs="+", help="PDF files and/or directories.")
    s.add_argument("--reset", action="store_true", help="Clear the index before ingesting.")
    s.set_defaults(func=cmd_ingest)

    s = sub.add_parser("ask", help="Ask a single question.")
    s.add_argument("question")
    s.add_argument("-k", type=int, default=config.TOP_K, help="Chunks to retrieve (default 4).")
    s.add_argument("--evaluate", action="store_true", help="Run the grounding evaluator.")
    s.add_argument("--json", action="store_true", help="Print JSON output.")
    s.set_defaults(func=cmd_ask)

    s = sub.add_parser("chat", help="Interactive Q&A session.")
    s.add_argument("-k", type=int, default=config.TOP_K)
    s.add_argument("--evaluate", action="store_true")
    s.set_defaults(func=cmd_chat)

    s = sub.add_parser("eval", help="Answer + grade a file of questions (.txt, .json, .jsonl).")
    s.add_argument("questions")
    s.add_argument("-k", type=int, default=config.TOP_K)
    s.add_argument("--out", help="Write detailed results to this JSON file.")
    s.add_argument("--include-contexts", action="store_true", help="Keep retrieved text in --out.")
    s.set_defaults(func=cmd_eval)

    sub.add_parser("stats", help="Show what is indexed.").set_defaults(func=cmd_stats)
    sub.add_parser("reset", help="Delete all indexed chunks.").set_defaults(func=cmd_reset)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (RuntimeError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
