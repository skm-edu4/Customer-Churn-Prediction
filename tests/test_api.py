"""FastAPI contract tests: health, model info, single and batch prediction."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from churn.models_io import save_sklearn_predictor, write_bundle
from churn.serving.api import create_app
from churn.training.baselines import make_pipeline

VALID = {
    "tenure": 2,
    "monthly_charges": 92.5,
    "total_charges": 185.0,
    "num_tickets": 3,
    "contract_type": "Month-to-month",
    "internet_service": "Fiber optic",
    "tech_support": "No",
    "payment_method": "Electronic check",
}


@pytest.fixture()
def client(tmp_path: Path, tiny_df) -> TestClient:
    """Build a tiny trained pipeline and serve it from a temp models dir."""
    X = tiny_df.drop(columns=["churned"])
    y = tiny_df["churned"]
    pipe = make_pipeline(LogisticRegression(max_iter=500))
    pipe.fit(X, y)

    models_dir = tmp_path / "models"
    entry = save_sklearn_predictor("logreg", pipe, models_dir)
    write_bundle(models_dir, "logreg", {"logreg": entry})

    app = create_app(models_dir)
    with TestClient(app) as c:
        yield c


def test_health(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model_loaded": True}


def test_model_info(client: TestClient):
    resp = client.get("/model-info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["best_model"] == "logreg"
    assert body["kind"] == "sklearn"
    assert body["available_models"] == ["logreg"]


def test_predict_returns_probability_and_band(client: TestClient):
    resp = client.post("/predict", json=VALID)
    assert resp.status_code == 200
    pred = resp.json()["prediction"]
    assert 0.0 <= pred["churn_probability"] <= 1.0
    assert pred["risk_band"] in {"low", "medium", "high"}
    assert pred["model"] == "logreg"


def test_predict_rejects_unknown_category(client: TestClient):
    bad = {**VALID, "contract_type": "Lifetime"}
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_predict_rejects_negative_charges(client: TestClient):
    bad = {**VALID, "monthly_charges": -5}
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_batch_prediction(client: TestClient):
    resp = client.post("/predict/batch", json={"customers": [VALID, {**VALID, "tenure": 48}]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    assert len(body["predictions"]) == 2


def test_missing_bundle_returns_503(tmp_path: Path):
    app = create_app(tmp_path / "does-not-exist")
    with TestClient(app) as c:
        assert c.get("/health").json()["model_loaded"] is False
        resp = c.post("/predict", json=VALID)
        assert resp.status_code == 503
