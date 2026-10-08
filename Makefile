PYTHON := uv run python

.PHONY: setup train evaluate report pipeline lint format typecheck test serve dashboard all clean

## one-time setup: create venv + install all dependencies
setup:
	uv sync

## train all model families and select the best one
train:
	$(PYTHON) scripts/train.py

## held-out evaluation + all figures (reports/figures)
evaluate:
	$(PYTHON) scripts/evaluate.py

## build reports/REPORT.md from the evaluation artefacts
report:
	$(PYTHON) scripts/generate_report.py

## train + evaluate + report
pipeline: train evaluate report

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

## everything a fresh clone needs to be green
all: lint typecheck test pipeline

## remove generated artefacts (keeps source + data untouched)
clean:
	rm -rf models reports mlruns data/processed
