"""Hand-label answers in the terminal, blind to the judge, to measure how much the LLM judge can be trusted.

    python -m eval.label eval/results/answers_<stamp>.json --n 40

Picks a fixed random sample (seed 0) of answerable questions, shows question, reference and answer, and asks
1 (correct) / 0 (incorrect) / s (skip) / q (save and quit). Labels go to eval/labels/hand_labels.csv; already
labelled ids are skipped, so it can be stopped and resumed.
"""

import argparse
import csv
import json
import random
import textwrap
from pathlib import Path

from eval.judge import LABELS, load_hand_labels

RULE = (
    "Correct = states the key fact of the reference and nothing that contradicts it. "
    "'I don't know' on an answerable question = incorrect."
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("answers", type=Path)
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument(
        "--fresh", action="store_true", help="skip questions already labelled in other runs (a clean held-out check)"
    )
    args = parser.parse_args()

    run = json.loads(args.answers.read_text(encoding="utf-8"))
    rows = [r for r in run["rows"] if r["answerable"]]
    if args.fresh and LABELS.exists():
        with open(LABELS, encoding="utf-8", newline="") as f:
            seen = {r["id"] for r in csv.DictReader(f) if r["answers_file"] != args.answers.name}
        rows = [r for r in rows if r["id"] not in seen]
    sample = random.Random(0).sample(rows, min(args.n, len(rows)))
    done = load_hand_labels(args.answers.name)
    todo = [r for r in sample if r["id"] not in done]
    print(f"{len(sample) - len(todo)} of {len(sample)} already labelled. {RULE}\n")

    LABELS.parent.mkdir(parents=True, exist_ok=True)
    new_file = not LABELS.exists()
    with open(LABELS, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label", "answers_file", "note"])
        if new_file:
            writer.writeheader()
        for n, r in enumerate(todo, start=1):
            print("=" * 100)
            print(f"[{n}/{len(todo)}] {r['id']}  Q: {r['question']}")
            print(textwrap.fill("REFERENCE: " + r["reference_answer"], 100))
            print(textwrap.fill("ANSWER:    " + r["answer"], 100))
            key = ""
            while key not in ("1", "0", "s", "q"):
                key = input("1 correct / 0 incorrect / s skip / q quit > ").strip().lower()
            if key == "q":
                break
            if key == "s":
                continue
            note = input("note (optional) > ").strip()
            writer.writerow({"id": r["id"], "label": key, "answers_file": args.answers.name, "note": note})
            f.flush()
    print(f"labels saved to {LABELS}")


if __name__ == "__main__":
    main()
