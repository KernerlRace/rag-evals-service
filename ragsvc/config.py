"""All configuration comes from environment variables, so docker compose, Helm and Terraform set the same names."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    store: str  # "postgres" or "memory" (memory reads data/index/size_<n>/ from disk)
    database_url: str
    chunk_size: int
    retriever: str  # bm25 | dense | hybrid | hybrid_rerank
    top_k: int
    llm_backend: str  # ollama | bedrock | extractive
    ollama_url: str
    ollama_model: str
    bedrock_model_id: str
    aws_region: str
    variant: str  # "stable" or "canary"; only used as a metrics label


def load_settings() -> Settings:
    env = os.environ.get
    return Settings(
        store=env("RAG_STORE", "postgres"),
        database_url=env("DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag"),
        chunk_size=int(env("RAG_CHUNK_SIZE", "256")),
        retriever=env("RAG_RETRIEVER", "hybrid"),
        top_k=int(env("RAG_TOP_K", "5")),
        llm_backend=env("RAG_LLM_BACKEND", "ollama"),
        ollama_url=env("OLLAMA_URL", "http://localhost:11434"),
        ollama_model=env("OLLAMA_MODEL", "qwen2.5:7b-instruct"),
        bedrock_model_id=env("BEDROCK_MODEL_ID", "eu.amazon.nova-micro-v1:0"),
        aws_region=env("AWS_REGION", "eu-west-1"),
        variant=env("RAG_VARIANT", "stable"),
    )
