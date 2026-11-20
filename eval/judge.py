"""LLM-as-judge for answer correctness, and how far to trust it.

    python -m eval.judge eval/results/answers_<stamp>.json --judge-model llama3.1:8b

* Answerable questions: the judge compares the answer to the reference answer -> correct / incorrect.
* Unanswerable questions: no judge needed, the answer is correct iff the model said "I don't know".
* If eval/labels/hand_labels.csv exists, the judge is scored against my hand labels (agreement, Cohen's kappa),
  and only then is its accuracy number worth reporting.

The judge is a different model family from the generator (Llama vs Qwen) to reduce self-preference bias.
"""

import argparse
import csv
import json
from pathlib import Path

from eval.common import fmt_ci, save_result
from eval.stats import bootstrap_ci, cohen_kappa
from ragsvc.llm import OllamaLLM

LABELS = Path(__file__).resolve().parent / "labels" / "hand_labels.csv"

JUDGE_SYSTEM = (
    "You are grading answers to questions about the Python programming language. "
    "You get a question, a reference answer written by an expert, and a candidate answer. "
    "The candidate is CORRECT if it states the key fact(s) of the reference answer and says nothing that "
    "contradicts it. Extra details are fine if they are not wrong. Different wording is fine. Citation markers "
    "like [1] are fine. The candidate is INCORRECT if it misses the key fact, contradicts the reference, or says "
    "it does not know. "
    'Reply with JSON only: {"reason": "<one short sentence>", "verdict": "correct" or "incorrect"}'
)


def judge_prompt(question: str, reference: str, candidate: str) -> str:
    return f"Question: {question}\n\nReference answer: {reference}\n\nCandidate answer: {candidate}"


def parse_verdict(text: str) -> int | None:
    try:
        verdict = str(json.loads(text).get("verdict", "")).strip().lower()
    except (json.JSONDecodeError, AttributeError):
        verdict = text.lower()
    if "incorrect" in verdict:
        return 0
    if "correct" in verdict:
        return 1
    return None


def load_hand_labels(path: Path = LABELS) -> dict[str, int]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8", newline="") as f:
        return {r["id"]: int(r["label"]) for r in csv.DictReader(f) if r["label"].strip() in ("0", "1")}


def agreement(judge: dict[str, int], hand: dict[str, int]) -> dict | None:
    ids = sorted(set(judge) & set(hand))
    if not ids:
        return None
    j = [judge[i] for i in ids]
    h = [hand[i] for i in ids]
    return {
        "n": len(ids),
        "agreement": bootstrap_ci([int(a == b) for a, b in zip(j, h, strict=True)]),
        "cohen_kappa": cohen_kappa(j, h),
        # rows = my label, columns = judge: how the judge errs matters (too lenient vs too strict)
        "confusion": {
            "hand1_judge1": sum(a == 1 and b == 1 for a, b in zip(h, j, strict=True)),
            "hand1_judge0": sum(a == 1 and b == 0 for a, b in zip(h, j, strict=True)),
            "hand0_judge1": sum(a == 0 and b == 1 for a, b in zip(h, j, strict=True)),
            "hand0_judge0": sum(a == 0 and b == 0 for a, b in zip(h, j, strict=True)),
        },
        "hand_accuracy_on_labelled": sum(h) / len(h),
        "judge_accuracy_on_labelled": sum(j) / len(j),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("answers", type=Path)
    parser.add_argument("--judge-model", default="llama3.1:8b")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    args = parser.parse_args()

    run = json.loads(args.answers.read_text(encoding="utf-8"))
    judge = OllamaLLM(args.ollama_url, args.judge_model)
    verdicts: dict[str, int] = {}
    reasons: dict[str, str] = {}
    for row in run["rows"]:
        if not row["answerable"]:
            verdicts[row["id"]] = int(row["abstained"])
            reasons[row["id"]] = "rule: unanswerable, correct iff abstained"
            continue
        out = judge.generate(judge_prompt(row["question"], row["reference_answer"], row["answer"]),
                             system=JUDGE_SYSTEM, json_mode=True)
        v = parse_verdict(out.text)
        verdicts[row["id"]] = 0 if v is None else v  # an unparseable verdict counts against the answer
        reasons[row["id"]] = out.text
        print(f"{row['id']} {'OK ' if v else 'BAD'} {out.text[:110]}")

    answerable = [r["id"] for r in run["rows"] if r["answerable"]]
    by_author = {}
    for author in sorted({r["author"] for r in run["rows"] if r["answerable"]}):
        ids = [r["id"] for r in run["rows"] if r["answerable"] and r["author"] == author]
        by_author[author] = bootstrap_ci([verdicts[i] for i in ids])
    hand = load_hand_labels()
    agree = agreement({i: verdicts[i] for i in answerable}, hand)
    result = {
        "answers_file": args.answers.name,
        "config": run["config"],
        "judge_model": args.judge_model,
        "judged_accuracy": bootstrap_ci([verdicts[i] for i in answerable]),
        "judged_accuracy_by_author": by_author,
        "unanswerable_correct": bootstrap_ci([verdicts[r["id"]] for r in run["rows"] if not r["answerable"]]),
        "judge_vs_hand": agree,
        "verdicts": verdicts,
        "reasons": reasons,
    }
    path = save_result("judge", result)
    print(f"saved {path}")
    print("judged accuracy (answerable):", fmt_ci(result["judged_accuracy"]))
    if agree:
        print(f"judge vs my labels on {agree['n']}: agreement {fmt_ci(agree['agreement'])}, "
              f"kappa {agree['cohen_kappa']:.2f}, confusion {agree['confusion']}")


if __name__ == "__main__":
    main()
