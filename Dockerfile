# syntax=docker/dockerfile:1
# One image, two services (api + dashboard) — docker-compose picks the command.
FROM ghcr.io/astral-sh/uv:0.11.8 AS uv

FROM python:3.11-slim AS base

# Reproducible, quiet, single-threaded BLAS (same guard the repo applies at
# import time — keeps tiny CPU containers predictable).
ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    OMP_NUM_THREADS=1

COPY --from=uv /uv /uvx /usr/local/bin/

WORKDIR /app

# --- dependency layer (cached unless pyproject/uv.lock change) -----------
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --no-install-project

# --- application layer ---------------------------------------------------
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY dashboard ./dashboard
RUN uv sync --frozen --no-dev

# --- runtime hardening ---------------------------------------------------
RUN useradd --create-home --shell /usr/sbin/nologin app \
    && chown -R app:app /app
USER app

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD /app/.venv/bin/python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

# Default command: the API (direct venv binaries — the runtime never needs uv).
# docker-compose overrides it for the dashboard service.
CMD ["/app/.venv/bin/uvicorn", "churn.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
