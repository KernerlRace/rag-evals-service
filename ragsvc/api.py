"""HTTP API: POST /ask, GET /health, GET /metrics."""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from ragsvc import metrics
from ragsvc.config import Settings, load_settings
from ragsvc.llm import make_llm
from ragsvc.pipeline import answer
from ragsvc.retrieval import METHODS, NumpyIndex, Retriever


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int | None = Field(default=None, ge=1, le=20)


class Source(BaseModel):
    chunk_id: str
    section: str
    url: str
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    retriever: str
    variant: str
    retrieval_ms: float
    generation_ms: float


def build_retriever(settings: Settings) -> Retriever:
    from ragsvc.models import Embedder, Reranker

    embedder = Embedder()
    reranker = Reranker() if settings.retriever == "hybrid_rerank" else None
    if settings.store == "postgres":
        from ragsvc import pgstore

        conn = pgstore.connect(settings.database_url)
        chunks = pgstore.load_chunks(conn)
        dense = pgstore.PgVectorIndex(conn)
    elif settings.store == "memory":
        from ragsvc import index

        chunks, vectors = index.load(settings.chunk_size)
        dense = NumpyIndex(vectors)
    else:
        raise ValueError(f"unknown RAG_STORE {settings.store!r}")
    if not chunks:
        raise RuntimeError("no chunks found: run `python -m ragsvc.ingest` first")
    return Retriever(chunks, embedder, dense, reranker)


def create_app(settings: Settings | None = None, retriever: Retriever | None = None, llm=None) -> FastAPI:
    settings = settings or load_settings()
    if settings.retriever not in METHODS:
        raise ValueError(f"RAG_RETRIEVER must be one of {METHODS}")
    state = {"retriever": retriever, "llm": llm}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Loading models and the BM25 index takes seconds, so it happens once at startup, not per request.
        if state["retriever"] is None:
            state["retriever"] = build_retriever(settings)
        if state["llm"] is None:
            state["llm"] = make_llm(settings)
        yield

    app = FastAPI(title="rag-evals-service", lifespan=lifespan)
    labels = {"variant": settings.variant, "retriever": settings.retriever}

    @app.get("/health")
    def health():
        r = state["retriever"]
        return {"status": "ok", "chunks": len(r.chunks) if r else 0, **labels}

    @app.get("/metrics")
    def prometheus_metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest):  # sync handler: FastAPI runs it in a thread pool, so slow LLM calls do not block
        t = time.perf_counter()
        try:
            result = answer(req.question, state["retriever"], state["llm"], settings.retriever, req.k or settings.top_k)
        except Exception as exc:
            metrics.REQUESTS.labels(**labels, status="error").inc()
            raise HTTPException(status_code=502, detail=f"generation failed: {type(exc).__name__}") from exc

        metrics.REQUESTS.labels(**labels, status="ok").inc()
        metrics.REQUEST_SECONDS.labels(**labels).observe(time.perf_counter() - t)
        metrics.RETRIEVAL_SECONDS.labels(**labels).observe(result.retrieval_seconds)
        metrics.GENERATION_SECONDS.labels(**labels).observe(result.generation_seconds)
        metrics.TOKENS.labels(**labels, kind="prompt").inc(result.generation.prompt_tokens)
        metrics.TOKENS.labels(**labels, kind="completion").inc(result.generation.completion_tokens)
        if result.abstained:
            metrics.ABSTAINED.labels(**labels).inc()

        return AskResponse(
            answer=result.text,
            sources=[Source(chunk_id=h.chunk.chunk_id, section=h.chunk.section, url=h.chunk.url, score=h.score)
                     for h in result.hits],
            retriever=settings.retriever,
            variant=settings.variant,
            retrieval_ms=round(result.retrieval_seconds * 1000, 1),
            generation_ms=round(result.generation_seconds * 1000, 1),
        )

    return app
