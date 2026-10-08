"""FastAPI service exposing the trained churn model.

Run locally::

    uvicorn churn.serving.api:app --reload

The service loads ``models/bundle.json`` at startup and always serves the
best-performing model from the last training run.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException

from churn.config import load_config
from churn.models_io import Predictor, load_bundle, load_predictor
from churn.serving.schemas import (
    BatchRequest,
    BatchResponse,
    CustomerRecord,
    HealthResponse,
    ModelInfo,
    Prediction,
    PredictResponse,
    risk_band,
)

API_VERSION = "2.0.0"


def _default_models_dir() -> Path:
    env = os.environ.get("CHURN_MODELS_DIR")
    if env:
        return Path(env)
    return load_config().models_dir


def create_app(bundle_dir: Path | None = None) -> FastAPI:
    models_dir = bundle_dir or _default_models_dir()
    state: dict[str, Predictor | None] = {"predictor": None}

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            state["predictor"] = load_predictor(models_dir)
        except Exception as exc:
            state["predictor"] = None
            print(f"[api] model loading failed: {exc}")
        yield

    app = FastAPI(
        title="Customer Churn Prediction API",
        description=(
            "Serves churn probabilities from the best model of the last training run. "
            "Sklearn pipelines, a PyTorch MLP and a TabTransformer are all available "
            "behind the same contract."
        ),
        version=API_VERSION,
        lifespan=lifespan,
    )

    def _predictor() -> Predictor:
        predictor = state["predictor"]
        if predictor is None:
            raise HTTPException(
                status_code=503,
                detail="No model loaded. Run `make train` first.",
            )
        return predictor

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health() -> HealthResponse:
        return HealthResponse(status="ok", model_loaded=state["predictor"] is not None)

    @app.get("/model-info", response_model=ModelInfo, tags=["ops"])
    def model_info() -> ModelInfo:
        try:
            bundle = load_bundle(models_dir)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        predictor = _predictor()
        return ModelInfo(
            best_model=bundle["best"],
            kind=predictor.kind,
            available_models=sorted(bundle["models"]),
            api_version=API_VERSION,
        )

    @app.post("/predict", response_model=PredictResponse, tags=["inference"])
    def predict(record: CustomerRecord) -> PredictResponse:
        predictor = _predictor()
        df = pd.DataFrame([record.model_dump()])
        prob = float(predictor.predict_proba(df)[0])
        return PredictResponse(
            prediction=Prediction(
                churn_probability=round(prob, 4),
                risk_band=risk_band(prob),
                model=predictor.name,
            )
        )

    @app.post("/predict/batch", response_model=BatchResponse, tags=["inference"])
    def predict_batch(request: BatchRequest) -> BatchResponse:
        predictor = _predictor()
        df = pd.DataFrame([c.model_dump() for c in request.customers])
        probs = predictor.predict_proba(df)
        return BatchResponse(
            model=predictor.name,
            count=len(probs),
            predictions=[
                Prediction(
                    churn_probability=round(float(p), 4),
                    risk_band=risk_band(float(p)),
                    model=predictor.name,
                )
                for p in probs
            ],
        )

    return app


# `uvicorn churn.serving.api:app` entrypoint
app = create_app()
