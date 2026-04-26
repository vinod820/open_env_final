FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV ENABLE_WEB_INTERFACE=false
ENV PORT=8000

COPY pyproject.toml README.md openenv.yaml ./
COPY social_engineer_arena ./social_engineer_arena
COPY scripts ./scripts
COPY assets ./assets
COPY outputs/evals ./outputs/evals

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch==2.5.1+cpu" && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -e ".[train]"

EXPOSE 8000
CMD ["python", "-m", "social_engineer_arena.server.app"]
