FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY tests ./tests
COPY scripts ./scripts
COPY .env.example .

RUN mkdir -p outputs images

EXPOSE 8000

# Shell form (with exec) so Render's $PORT expands at container start; the
# 8000 fallback keeps `docker compose up` and plain `docker run` unchanged.
# `exec` replaces the shell so uvicorn receives SIGTERM directly.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
