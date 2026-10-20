# One image, deployable two ways:
#
#   * default CMD  → FastAPI + Streamlit in ONE container on port 7860
#                    (Hugging Face Spaces runs a single container; see
#                    scripts/deploy_entrypoint.sh)
#   * docker compose overrides the command per service (api on 8000,
#     dashboard on 8501) and bind-mounts ./models over the baked ones —
#     retraining stays a host-side `make train`, no rebuild needed.
#
# Trained artefacts are BAKED IN at build time (seed-reproducible), so the
# image runs standalone with zero mounts: `docker run -p 7860:7860 <image>`.
FROM ghcr.io/astral-sh/uv:0.11.8 AS uv

FROM python:3.11-slim AS base

# Reproducible, quiet, single-threaded BLAS (same guard the repo applies at
# import time — keeps tiny CPU containers predictable; also guarantees the
# baked model matches the published numbers bit-for-bit).
# UV_NO_CACHE: never bake uv's download cache into a layer (it duplicates
# the whole venv). Bytecode compilation stays on for fast cold starts.
ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    OMP_NUM_THREADS=1

COPY --from=uv /uv /uvx /usr/local/bin/

WORKDIR /app

# Create the runtime user BEFORE any large file exists, so every artefact
# below is born app-owned. (A post-hoc `chown -R` over a populated /app
# duplicates it entirely in a separate copy-on-write layer.) Just the
# directory itself here — it's still empty, so this costs nothing.
RUN useradd --create-home --shell /usr/sbin/nologin app \
    && chown app:app /app

# --- dependency layer (cached unless pyproject/uv.lock change) -----------
COPY --chown=app:app pyproject.toml uv.lock README.md LICENSE ./

# torch resolves from the official CPU wheel index (see pyproject [tool.uv]),
# so no CUDA stack comes from torch. xgboost's Linux wheels still declare
# nvidia-nccl (GPU distributed training, never imported on a CPU host) —
# exclude whatever GPU-only packages the lock references; the list is derived
# from uv.lock itself so it stays correct when dependencies bump.
# (`--no-install-package` skips installation without touching resolution.)
USER app
RUN set -- \
    && for p in $(grep -oE 'name = "(nvidia-[^"]+|triton)"' uv.lock \
        | sed 's/name = "//; s/"//' | sort -u); do set -- "$@" --no-install-package "$p"; done \
    && echo "excluding GPU-only packages:$*" \
    && uv sync --frozen --no-dev --no-install-project "$@"

# --- application layer ---------------------------------------------------
COPY --chown=app:app src ./src
COPY --chown=app:app scripts ./scripts
COPY --chown=app:app configs ./configs
COPY --chown=app:app dashboard ./dashboard
RUN set -- \
    && for p in $(grep -oE 'name = "(nvidia-[^"]+|triton)"' uv.lock \
        | sed 's/name = "//; s/"//' | sort -u); do set -- "$@" --no-install-package "$p"; done \
    && uv sync --frozen --no-dev "$@"

# --- bake trained artefacts (seed-reproducible, matches published numbers) --
# Full pipeline: train → evaluate → benchmark → report. Runs as `app`.
# .dockerignore keeps stale host artefacts out of the build context — what
# gets baked is always generated fresh from configs/default.yaml.
RUN /app/.venv/bin/python scripts/train.py \
    && /app/.venv/bin/python scripts/evaluate.py \
    && /app/.venv/bin/python scripts/benchmark.py \
    && /app/.venv/bin/python scripts/generate_report.py

# Port 7860 is the platform contract for the combined container; compose
# maps 8000/8501 explicitly per service.
EXPOSE 7860 8000 8501

# Health of the public surface (the combined CMD's Streamlit). compose
# overrides this per service (api → /health, dashboard → /_stcore/health).
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD /app/.venv/bin/python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/_stcore/health', timeout=4)"

# Default: both services in one container, Streamlit is PID 1 (public).
CMD ["/app/scripts/deploy_entrypoint.sh"]
