"""Decide whether the canary retriever config should be promoted, by comparing it with stable in Prometheus.

    python -m ops.canary_check --prometheus http://localhost:30090 --window 15m

Checks, all on the same time window and live traffic split by the Service:
  1. enough canary traffic to say anything (>= --min-requests answered requests)
  2. error rate:      canary <= stable + 1 percentage point
  3. p95 latency:     canary <= 1.25 x stable + 0.5 s   (the reranker is expected to cost some latency)
  4. "I don't know":  canary <= stable + 10 percentage points (a retriever that finds less makes the LLM abstain)
Quality itself is not judged from live traffic (there are no labels there); it is gated offline by the eval
results before the canary is deployed. Exit code: 0 promote, 1 roll back, 2 not enough data yet.
"""

import argparse
import sys
from dataclasses import dataclass

import httpx


@dataclass
class VariantStats:
    requests: float
    error_rate: float
    p95_seconds: float
    abstain_rate: float


def decide(stable: VariantStats, canary: VariantStats, min_requests: int = 30) -> tuple[str, list[str]]:
    if canary.requests < min_requests:
        return "WAIT", [f"only {canary.requests:.0f} canary requests, need {min_requests}"]
    failures = []
    if canary.error_rate > stable.error_rate + 0.01:
        failures.append(f"error rate {canary.error_rate:.1%} vs stable {stable.error_rate:.1%}")
    if canary.p95_seconds > 1.25 * stable.p95_seconds + 0.5:
        failures.append(f"p95 {canary.p95_seconds:.2f}s vs stable {stable.p95_seconds:.2f}s")
    if canary.abstain_rate > stable.abstain_rate + 0.10:
        failures.append(f"abstain rate {canary.abstain_rate:.1%} vs stable {stable.abstain_rate:.1%}")
    return ("ROLLBACK", failures) if failures else ("PROMOTE", ["all checks passed"])


def query(prom: str, expr: str) -> dict[str, float]:
    r = httpx.get(f"{prom}/api/v1/query", params={"query": expr}, timeout=10)
    r.raise_for_status()
    out = {}
    for series in r.json()["data"]["result"]:
        value = float(series["value"][1])
        out[series["metric"].get("variant", "")] = 0.0 if value != value else value  # NaN (0/0) -> 0
    return out


def collect(prom: str, window: str) -> dict[str, VariantStats]:
    total = query(prom, f"sum by (variant) (increase(rag_requests_total[{window}]))")
    ok = query(prom, f'sum by (variant) (increase(rag_requests_total{{status="ok"}}[{window}]))')
    errors = query(prom, f'sum by (variant) (increase(rag_requests_total{{status="error"}}[{window}]))')
    abstained = query(prom, f"sum by (variant) (increase(rag_abstained_total[{window}]))")
    p95 = query(prom, f"histogram_quantile(0.95, sum by (le, variant) (rate(rag_request_seconds_bucket[{window}])))")
    stats = {}
    for v in total:
        n_ok = ok.get(v, 0.0)
        stats[v] = VariantStats(
            requests=n_ok,
            error_rate=errors.get(v, 0.0) / total[v] if total[v] else 0.0,
            p95_seconds=p95.get(v, 0.0),
            abstain_rate=abstained.get(v, 0.0) / n_ok if n_ok else 0.0,
        )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prometheus", default="http://localhost:30090")
    parser.add_argument("--window", default="15m")
    parser.add_argument("--min-requests", type=int, default=30)
    args = parser.parse_args()

    stats = collect(args.prometheus, args.window)
    for variant, s in sorted(stats.items()):
        print(f"{variant:7s} requests {s.requests:5.0f}  errors {s.error_rate:6.1%}  p95 {s.p95_seconds:6.2f}s  "
              f"abstained {s.abstain_rate:6.1%}")
    if "stable" not in stats or "canary" not in stats:
        print("WAIT: need traffic on both stable and canary")
        sys.exit(2)
    verdict, reasons = decide(stats["stable"], stats["canary"], args.min_requests)
    print(f"{verdict}: " + "; ".join(reasons))
    sys.exit({"PROMOTE": 0, "ROLLBACK": 1, "WAIT": 2}[verdict])


if __name__ == "__main__":
    main()
