FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV ENABLE_WEB_INTERFACE=true
ENV PORT=8000

COPY pyproject.toml README.md openenv.yaml ./
COPY social_engineer_arena ./social_engineer_arena

RUN pip install --no-cache-dir -e .

EXPOSE 8000
CMD ["python", "-m", "social_engineer_arena.server.app"]
