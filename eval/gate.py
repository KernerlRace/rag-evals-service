"""Retrieval regression gate for CI: fail the build if retrieval quality drops.

    python -m eval.gate            # check against eval/gate_baseline.json
    python -m eval.gate --update   # accept the current numbers as the new baseline (commit the file with a reason)

It runs BM25 only, because BM25 needs no model download and is deterministic, so CI stays fast (seconds) and a
change in the numbers always means a change in the code or data. What it protects is everything every retriever
depends on: document loading, section splitting, chunking, tokenisation, the golden set and the relevance rule.
Those are the parts that break silently. A unit test still passes when a filter drops a whole folder of the docs;
this gate does not. That is the first bug this project had (built-ins missing from the corpus).

The dense and reranked numbers need the ONNX models and ~15 minutes of CPU, so they stay in `eval.retrieval_eval`.
"""

import argparse
import json
import sys
from pathlib import Path

from eval import golden
from eval.stats import bootstrap_ci
from ragsvc.bm25 import BM25
from ragsvc.chunking import chunk_corpus
from ragsvc.corpus import load_documents

BASELINE = Path(__file__).resolve().parent / "gate_baseline.json"
CHUNK_SIZE = 256
TOLERANCE = 0.02  # golden-set edits move the numbers a little; a real regression moves them a lot


def measure() -> dict:
    docs = load_documents()
    items, problems = golden.locate(golden.load(), docs)
    if problems:
        sys.exit("golden passages not found in the docs:\n" + "\n".join(problems))
    items = [it for it in items if it.answerable]
    chunks = chunk_corpus(list(docs.values()), CHUNK_SIZE)
    bm25 = BM25([c.text for c in chunks])
    ranks = [golden.first_relevant_rank([chunks[i] for i, _ in bm25.search(it.question, 10)], it) for it in items]
    hit5 = [int(r is not None and r <= 5) for r in ranks]
    mrr = [1.0 / r if r is not None else 0.0 for r in ranks]
    return {
        "method": "bm25",
        "chunk_size": CHUNK_SIZE,
        "n_questions": len(items),
        "n_chunks": len(chunks),
        "hit@5": round(bootstrap_ci(hit5)["mean"], 4),
        "mrr@10": round(bootstrap_ci(mrr)["mean"], 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--update", action="store_true", help="write the current numbers as the new baseline")
    args = parser.parse_args()

    now = measure()
    print(json.dumps(now))
    if args.update:
        BASELINE.write_text(json.dumps(now, indent=2) + "\n", encoding="utf-8")
        print(f"baseline written to {BASELINE.name}")
        return

    base = json.loads(BASELINE.read_text(encoding="utf-8"))
    failures = [
        f"{m}: {now[m]:.3f} < baseline {base[m]:.3f} - {TOLERANCE}"
        for m in ("hit@5", "mrr@10")
        if now[m] < base[m] - TOLERANCE
    ]
    if now["n_questions"] < base["n_questions"]:
        failures.append(f"golden set shrank: {now['n_questions']} < {base['n_questions']} questions")
    if failures:
        sys.exit("retrieval regression:\n  " + "\n  ".join(failures))
    print(f"ok: hit@5 {now['hit@5']:.3f} (baseline {base['hit@5']:.3f}), mrr@10 {now['mrr@10']:.3f}")


if __name__ == "__main__":
    main()
