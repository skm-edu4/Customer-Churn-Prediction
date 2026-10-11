PYTHON := uv run python

.PHONY: setup train evaluate report pipeline hpo benchmark lint format typecheck test serve dashboard docker-build docker-up docker-down all clean

## one-time setup: create venv + install all dependencies
setup:
	uv sync

## train all model families and select the best one
train:
	$(PYTHON) scripts/train.py

## Optuna HPO run: tunes the classical families, then trains as usual
hpo:
	$(PYTHON) scripts/train.py --hpo

## held-out evaluation + all figures (reports/figures)
evaluate:
	$(PYTHON) scripts/evaluate.py

## ONNX export + parity check + latency benchmark (reports/benchmark.md)
benchmark:
	$(PYTHON) scripts/benchmark.py

## build reports/REPORT.md from the evaluation artefacts
report:
	$(PYTHON) scripts/generate_report.py

## train + evaluate + benchmark + report
pipeline: train evaluate benchmark report

## static analysis
lint:
	uv run ruff check src tests scripts

format:
	uv run ruff format src tests scripts

typecheck:
	uv run mypy src/churn

## unit + integration tests
test:
	uv run pytest

## start the FastAPI service (http://localhost:8000/docs)
serve:
	uv run uvicorn churn.serving.api:app --reload

## start the Streamlit dashboard (http://localhost:8501)
dashboard:
	uv run streamlit run dashboard/app.py

## build the container image
docker-build:
	docker compose build

## run API + dashboard in containers (requires `make pipeline` first)
docker-up:
	docker compose up -d --build

docker-down:
	docker compose down

## everything a fresh clone needs to be green
all: lint typecheck test pipeline

## remove generated artefacts (keeps source + data untouched)
clean:
	rm -rf models reports mlruns data/processed
