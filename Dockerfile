FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV ENABLE_WEB_INTERFACE=false
ENV PORT=8000
ENV PYTHONPATH=/app

COPY pyproject.toml openenv.yaml ./

# Install runtime + training dependencies in a cache-friendly layer.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch==2.5.1+cpu" && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
      "openenv-core>=0.1.0" "fastapi>=0.104.0" "uvicorn>=0.24.0" "pydantic>=2.0.0" \
      "trl>=0.18.0" "transformers>=4.44.0" "datasets>=2.20.0" "accelerate>=0.33.0"

COPY README.md ./
COPY social_engineer_arena ./social_engineer_arena
COPY scripts ./scripts
COPY assets ./assets
COPY outputs/evals ./outputs/evals

EXPOSE 8000
CMD ["python", "-m", "social_engineer_arena.server.app"]
