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
    parser.add_argument("--pause", type=float, default=0.0, help="seconds each worker waits between requests")
    args = parser.parse_args()

    questions = [it.question for it in golden.load()] * args.rounds
    random.Random(0).shuffle(questions)
    # A new TCP connection per request: kube-proxy balances connections, not requests, so one keep-alive
    # connection would send every request to the same pod and the canary would never see traffic.
    client = httpx.Client(base_url=args.url, timeout=180, headers={"Connection": "close"})
    seen = Counter()

    def ask(q: str) -> None:
        try:
            r = client.post("/ask", json={"question": q})
            variant = r.json().get("variant", "?") if r.status_code == 200 else f"http {r.status_code}"
        except httpx.TransportError as e:  # a pod dying mid-request is a result to count, not a reason to stop
            variant = type(e).__name__
        seen[variant] += 1
        time.sleep(args.pause)  # keeps the GPU below 100% duty cycle on a desktop card

    t = time.perf_counter()
    with ThreadPoolExecutor(args.workers) as pool:
        for n, _ in enumerate(pool.map(ask, questions), start=1):
            if n % 20 == 0:
                print(f"{n}/{len(questions)} {dict(seen)}")
    print(f"done in {time.perf_counter() - t:.0f}s: {dict(seen)}")


if __name__ == "__main__":
    main()
