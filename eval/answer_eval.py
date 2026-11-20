"""Generate answers for every golden question with one pipeline config and save them, with latencies and tokens.

    python -m eval.answer_eval --method hybrid_rerank --size 256 --backend ollama --model qwen2.5:7b-instruct
    python -m eval.answer_eval --backend extractive      # no LLM: first sentence of the top passage (floor)

Judging happens in a separate step (eval.judge), so the same answers can be re-judged without regenerating.
"""

import argparse
import dataclasses

from eval import golden
from eval.common import save_result
from eval.stats import bootstrap_ci, percentiles
from ragsvc import index
from ragsvc.config import load_settings
from ragsvc.corpus import load_documents
from ragsvc.llm import make_llm
from ragsvc.pipeline import answer
from ragsvc.retrieval import METHODS, NumpyIndex, Retriever


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", default="hybrid_rerank", choices=METHODS)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--backend", default="ollama", choices=["ollama", "bedrock", "extractive"])
    parser.add_argument("--model", default=None, help="Ollama model or Bedrock model id")
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions (smoke test)")
    args = parser.parse_args()

    from ragsvc.models import Embedder, Reranker

    settings = load_settings()
    if args.model and args.backend == "ollama":
        settings = dataclasses.replace(settings, ollama_model=args.model)
    if args.model and args.backend == "bedrock":
        settings = dataclasses.replace(settings, bedrock_model_id=args.model)
    settings = dataclasses.replace(settings, llm_backend=args.backend)
    llm = make_llm(settings)
    model_name = {"ollama": settings.ollama_model, "bedrock": settings.bedrock_model_id}.get(args.backend, "none")

    items, problems = golden.locate(golden.load(), load_documents())
    if problems:
        raise SystemExit("\n".join(problems))
    items = items[: args.limit]

    embedder = Embedder()
    chunks, vectors = index.load_or_build(args.size, embedder)
    retriever = Retriever(chunks, embedder, NumpyIndex(vectors), Reranker() if args.method == "hybrid_rerank" else None)

    llm.generate("Say OK.")  # warm-up: the first Ollama call loads the model into VRAM and would skew latency
    rows = []
    for n, it in enumerate(items, start=1):
        a = answer(it.question, retriever, llm, args.method, args.k)
        rank = golden.first_relevant_rank([h.chunk for h in a.hits], it) if it.answerable else None
        rows.append({
            "id": it.id, "question": it.question, "reference_answer": it.reference_answer,
            "answerable": it.answerable, "author": it.author, "answer": a.text, "abstained": a.abstained,
            "gold_rank": rank, "sources": [h.chunk.chunk_id for h in a.hits],
            "retrieval_s": a.retrieval_seconds, "generation_s": a.generation_seconds,
            "prompt_tokens": a.generation.prompt_tokens, "completion_tokens": a.generation.completion_tokens,
        })
        print(f"[{n}/{len(items)}] {it.id} {a.generation_seconds:5.1f}s  {a.text[:90]!r}")

    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    summary = {
        "latency_total": percentiles([r["retrieval_s"] + r["generation_s"] for r in rows]),
        "latency_retrieval": percentiles([r["retrieval_s"] for r in rows]),
        "latency_generation": percentiles([r["generation_s"] for r in rows]),
        "mean_prompt_tokens": sum(r["prompt_tokens"] for r in rows) / len(rows),
        "mean_completion_tokens": sum(r["completion_tokens"] for r in rows) / len(rows),
        # Abstaining on an answerable question is a miss; on an unanswerable one it is the right answer.
        "false_abstention_rate": bootstrap_ci([r["abstained"] for r in answerable]) if answerable else None,
        "correct_abstention_rate": bootstrap_ci([r["abstained"] for r in unanswerable]) if unanswerable else None,
    }
    path = save_result("answers", {
        "config": {"method": args.method, "size": args.size, "k": args.k, "backend": args.backend,
                   "model": model_name},
        "summary": summary, "rows": rows,
    })
    print(f"saved {path}")
    print({k: v for k, v in summary.items() if k.startswith("latency")})


if __name__ == "__main__":
    main()
