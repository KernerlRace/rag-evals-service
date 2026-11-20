"""Prometheus metrics. Every series carries `variant` and `retriever` so stable and canary can be compared."""

from prometheus_client import Counter, Histogram

LABELS = ("variant", "retriever")
_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 40)

REQUEST_SECONDS = Histogram("rag_request_seconds", "End-to-end /ask latency", LABELS, buckets=_BUCKETS)
RETRIEVAL_SECONDS = Histogram("rag_retrieval_seconds", "Retrieval latency (all stages)", LABELS, buckets=_BUCKETS)
GENERATION_SECONDS = Histogram("rag_generation_seconds", "LLM generation latency", LABELS, buckets=_BUCKETS)
REQUESTS = Counter("rag_requests_total", "/ask requests by outcome", (*LABELS, "status"))
TOKENS = Counter("rag_llm_tokens_total", "LLM tokens", (*LABELS, "kind"))
ABSTAINED = Counter("rag_abstained_total", "Answers that were exactly \"I don't know\"", LABELS)
