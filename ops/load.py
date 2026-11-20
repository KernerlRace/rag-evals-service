"""Send the golden-set questions to a running service to create traffic for the dashboards and the canary check.

    python -m ops.load --url http://localhost:30080 --rounds 2 --workers 2
"""

import argparse
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import httpx

from eval import golden


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:30080")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)  # one GPU behind Ollama: more workers only queue
    args = parser.parse_args()

    questions = [it.question for it in golden.load()] * args.rounds
    random.Random(0).shuffle(questions)
    client = httpx.Client(base_url=args.url, timeout=180)
    seen = Counter()

    def ask(q: str) -> None:
        r = client.post("/ask", json={"question": q})
        variant = r.json().get("variant", "?") if r.status_code == 200 else f"http {r.status_code}"
        seen[variant] += 1

    t = time.perf_counter()
    with ThreadPoolExecutor(args.workers) as pool:
        for n, _ in enumerate(pool.map(ask, questions), start=1):
            if n % 20 == 0:
                print(f"{n}/{len(questions)} {dict(seen)}")
    print(f"done in {time.perf_counter() - t:.0f}s: {dict(seen)}")


if __name__ == "__main__":
    main()
