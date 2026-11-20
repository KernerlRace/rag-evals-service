# One image for the API and the ingestion job. ONNX Runtime instead of torch keeps it well under 1 GB.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.7 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH FASTEMBED_CACHE_PATH=/models PYTHONUNBUFFERED=1

WORKDIR /app
# Dependencies first, so code changes do not reinstall them (layer cache).
COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --locked --no-dev --no-install-project --extra bedrock
COPY ragsvc ragsvc
COPY data/raw data/raw
RUN uv sync --locked --no-dev --extra bedrock

# Bake both models into the image: pods start without downloading anything and work offline.
RUN python -c "from ragsvc.models import Embedder, Reranker; Embedder(); Reranker()" \
    && chmod -R a+rX /models && mkdir -p /app/data/index && chown -R 65534:65534 /app/data/index

USER 65534
EXPOSE 8000
CMD ["uvicorn", "ragsvc.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
