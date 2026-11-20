"""Retrieval ablations: BM25 vs dense vs hybrid vs hybrid + reranker, at several chunk sizes.

    python -m eval.retrieval_eval                      # all methods, chunk sizes 128/256/512
    python -m eval.retrieval_eval --sizes 256 --methods bm25 hybrid

Metrics per question (then averaged, with bootstrap 95% CIs over questions):
  hit@k     1 if a relevant chunk is in the top k (= recall@k, since each question has one gold passage)
  MRR@10    1 / rank of the first relevant chunk, 0 if it is not in the top 10
  hit@budget  hit within a fixed context budget of ~1024 words (k = 1024 // chunk size). Plain hit@k favours
            big chunks, because a 512-word chunk covers more text than a 128-word one; the budget makes the
            comparison fair: same amount of text handed to the LLM.
"""

import argparse
import time

import numpy as np

from eval import golden
from eval.common import fmt_ci, save_result
from eval.stats import bootstrap_ci, paired_bootstrap_diff, percentiles
from ragsvc import index
from ragsvc.corpus import load_documents
from ragsvc.retrieval import METHODS, NumpyIndex, Retriever

KS = (1, 3, 5, 10)
BUDGET_WORDS = 1024


def evaluate(retriever: Retriever, items: list[golden.Item], method: str, budget_k: int) -> dict:
    depth = max(max(KS), budget_k)
    ranks, seconds = [], []
    for it in items:
        t = time.perf_counter()
        result = retriever.search(it.question, method=method, k=depth)
        seconds.append(time.perf_counter() - t)
        ranks.append(golden.first_relevant_rank([h.chunk for h in result.hits], it))

    per_q = {f"hit@{k}": [int(r is not None and r <= k) for r in ranks] for k in KS}
    per_q["hit@budget"] = [int(r is not None and r <= budget_k) for r in ranks]
    per_q["mrr@10"] = [1.0 / r if r is not None and r <= 10 else 0.0 for r in ranks]
    return {
        "metrics": {name: bootstrap_ci(vals) for name, vals in per_q.items()},
        "latency": percentiles(seconds),
        "per_question": [{"id": it.id, "rank": r} for it, r in zip(items, ranks, strict=True)],
        "_raw": per_q,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    parser.add_argument("--author", default=None, help="only questions by this author (draft / mykyta)")
    args = parser.parse_args()

    from ragsvc.models import EMBED_MODEL, RERANK_MODEL, Embedder, Reranker

    items, problems = golden.locate(golden.load(), load_documents())
    if problems:
        raise SystemExit("golden set problems:\n" + "\n".join(problems))
    items = [it for it in items if it.answerable and (args.author is None or it.author == args.author)]
    print(f"{len(items)} answerable questions")

    embedder = Embedder()
    reranker = Reranker() if "hybrid_rerank" in args.methods else None
    runs = []
    for size in args.sizes:
        chunks, vectors = index.load_or_build(size, embedder)
        retriever = Retriever(chunks, embedder, NumpyIndex(vectors), reranker)
        budget_k = max(1, BUDGET_WORDS // size)
        for method in args.methods:
            run = evaluate(retriever, items, method, budget_k)
            run.update(size=size, method=method, n_chunks=len(chunks), budget_k=budget_k)
            runs.append(run)
            m = run["metrics"]
            print(f"size {size:4d} {method:14s} hit@5 {fmt_ci(m['hit@5'])}  mrr {fmt_ci(m['mrr@10'])}  "
                  f"hit@budget(k={budget_k}) {fmt_ci(m['hit@budget'])}  p50 {run['latency']['p50_ms']:.0f} ms")

    # Paired comparisons on the same questions: every method against BM25 at the same chunk size.
    comparisons = []
    by_key = {(r["size"], r["method"]): r for r in runs}
    for (size, method), run in by_key.items():
        base = by_key.get((size, "bm25"))
        if base is None or method == "bm25":
            continue
        for metric in ("hit@5", "mrr@10"):
            diff = paired_bootstrap_diff(run["_raw"][metric], base["_raw"][metric])
            comparisons.append({"size": size, "a": method, "b": "bm25", "metric": metric, **diff})
    # Chunk size, at equal context budget, for each method against 256 words.
    for method in args.methods:
        ref = by_key.get((256, method))
        for size in args.sizes:
            run = by_key.get((size, method))
            if ref is None or run is None or size == 256:
                continue
            diff = paired_bootstrap_diff(run["_raw"]["hit@budget"], ref["_raw"]["hit@budget"])
            comparisons.append({"method": method, "a": f"size {size}", "b": "size 256", "metric": "hit@budget", **diff})

    for run in runs:
        del run["_raw"]
    path = save_result("retrieval", {
        "embed_model": EMBED_MODEL,
        "rerank_model": RERANK_MODEL,
        "n_questions": len(items),
        "author_filter": args.author,
        "budget_words": BUDGET_WORDS,
        "runs": runs,
        "comparisons": comparisons,
        "seed": 0,
        "numpy": np.__version__,
    })
    print(f"saved {path}")
    for c in comparisons:
        flag = "*" if c["significant"] else " "
        print(f"{flag} {c.get('size', c.get('method'))} {c['a']} - {c['b']} {c['metric']}: {fmt_ci(c)}")


if __name__ == "__main__":
    main()
