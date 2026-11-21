"""Score every saved judge run of one answers file against the hand labels for that file. Offline: no LLM calls.

    python -m eval.judge_compare eval/results/answers_<stamp>.json
"""

import argparse
import json
from pathlib import Path

from eval.common import fmt_ci
from eval.judge import agreement, load_hand_labels

RESULTS = Path(__file__).resolve().parent / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("answers", type=Path)
    args = parser.parse_args()

    hand = load_hand_labels(args.answers.name)
    print(f"{len(hand)} hand labels for {args.answers.name}: {sum(hand.values())} correct, "
          f"{len(hand) - sum(hand.values())} incorrect")
    for path in sorted(RESULTS.glob("judge_*.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        if run["answers_file"] != args.answers.name:
            continue
        a = agreement(run["verdicts"], hand)
        if a is None:
            continue
        c = a["confusion"]
        print(f"{path.name}  prompt {run.get('judge_prompt', 'v1')}  accuracy {fmt_ci(run['judged_accuracy'])}  "
              f"agreement {fmt_ci(a['agreement'])}  kappa {a['cohen_kappa']:.2f}  "
              f"wrong answers caught {c['hand0_judge0']}/{c['hand0_judge0'] + c['hand0_judge1']}  "
              f"right answers rejected {c['hand1_judge0']}/{c['hand1_judge0'] + c['hand1_judge1']}")


if __name__ == "__main__":
    main()
