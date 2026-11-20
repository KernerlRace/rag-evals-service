import dataclasses

from fastapi.testclient import TestClient

from ragsvc.api import create_app
from ragsvc.config import load_settings
from ragsvc.llm import ExtractiveLLM, build_prompt


class BrokenLLM:
    def generate(self, prompt, **kwargs):
        raise ConnectionError("ollama is down")


def make_client(retriever, llm=None):
    settings = dataclasses.replace(load_settings(), retriever="hybrid_rerank", variant="canary")
    return TestClient(create_app(settings, retriever=retriever, llm=llm or ExtractiveLLM()))


def test_health(toy_retriever):
    with make_client(toy_retriever) as client:
        body = client.get("/health").json()
    assert body == {"status": "ok", "chunks": len(toy_retriever.chunks), "variant": "canary",
                    "retriever": "hybrid_rerank"}


def test_ask_returns_answer_with_sources(toy_retriever):
    with make_client(toy_retriever) as client:
        r = client.post("/ask", json={"question": "Where is the smallest element of a heap?", "k": 2})
    assert r.status_code == 200
    body = r.json()
    assert "heap[0]" in body["answer"]
    assert body["sources"][0]["url"] == "https://docs.python.org/3.13/library/heapq.html"
    assert len(body["sources"]) == 2


def test_ask_validates_input(toy_retriever):
    with make_client(toy_retriever) as client:
        assert client.post("/ask", json={"question": ""}).status_code == 422
        assert client.post("/ask", json={"question": "valid question", "k": 0}).status_code == 422


def test_llm_failure_is_a_502_and_counted(toy_retriever):
    with make_client(toy_retriever, llm=BrokenLLM()) as client:
        r = client.post("/ask", json={"question": "what is a heap?"})
        assert r.status_code == 502
        metrics = client.get("/metrics").text
    assert 'rag_requests_total{retriever="hybrid_rerank",status="error",variant="canary"}' in metrics


def test_metrics_expose_latency_histograms(toy_retriever):
    with make_client(toy_retriever) as client:
        client.post("/ask", json={"question": "json keys type"})
        text = client.get("/metrics").text
    for name in ("rag_request_seconds_bucket", "rag_retrieval_seconds_count", "rag_llm_tokens_total"):
        assert name in text


def test_prompt_numbers_passages():
    prompt = build_prompt("q?", ["first", "second"])
    assert "[1] first" in prompt and "[2] second" in prompt and prompt.endswith("Answer:")
