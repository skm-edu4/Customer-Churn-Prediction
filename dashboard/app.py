"""Streamlit dashboard: EDA, model comparison, explainability, live risk scoring.

Run with::

    make dashboard          # (API optional — falls back to local model)

The dashboard calls the FastAPI service when it is reachable and loads the
model directly otherwise, so a recruiter can explore results with or without
``make serve`` running.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from churn.config import load_config  # noqa: E402

st.set_page_config(
    page_title="Churn Prediction — Model Dashboard",
    page_icon="📉",
    layout="wide",
)

API_URL = os.environ.get("CHURN_API_URL", "http://localhost:8000")

CONTRACTS = ["Month-to-month", "One year", "Two year"]
INTERNET = ["Fiber optic", "DSL", "No internet"]
YESNO = ["No", "Yes"]
PAYMENTS = ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]


@st.cache_resource
def _load_context():
    """Load config, results and (lazily) a local predictor fallback."""
    cfg = load_config()
    results_path = cfg.models_dir / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else None
    return cfg, results


def _predict_via_api(payload: dict) -> dict | None:
    import urllib.error
    import urllib.request

    req = urllib.request.Request(
        f"{API_URL}/predict",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return None


def _predict_local(payload: dict) -> dict | None:
    try:
        import pandas as pd

        from churn.models_io import load_predictor
        from churn.serving.schemas import risk_band

        cfg, _ = _load_context()
        predictor = st.session_state.setdefault("_predictor", None) or load_predictor(
            cfg.models_dir
        )
        st.session_state["_predictor"] = predictor
        prob = float(predictor.predict_proba(pd.DataFrame([payload]))[0])
        return {
            "prediction": {
                "churn_probability": round(prob, 4),
                "risk_band": risk_band(prob),
                "model": predictor.name,
            }
        }
    except Exception:
        return None


def _fig(key: str, caption: str) -> None:
    cfg, results = _load_context()
    rel = (results or {}).get("figures", {}).get(key)
    path = cfg.project_root / rel if rel else None
    if path and path.exists():
        st.image(str(path), caption=caption, use_container_width=True)


def _table(rows: list[dict], columns: list[str]):
    import pandas as pd

    return pd.DataFrame(rows).set_index("model")[columns] if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
cfg, results = _load_context()

st.title("📉 Customer Churn — Model Dashboard")

if results is None:
    st.warning(
        "No results found. Run the pipeline first:\n\n"
        "```bash\nmake pipeline\n```"
    )
    st.stop()

best = results["best"]
dataset = results["dataset"]

st.caption(
    f"dataset `{dataset['source']}` · {dataset['rows']} rows · "
    f"churn rate {dataset['churn_rate']:.1%} · seed {results['seed']} · "
    f"**best model: `{best}`**"
)

tab_overview, tab_models, tab_explain, tab_score = st.tabs(
    ["🔎 Overview", "🏆 Model comparison", "🧪 Explainability", "🎯 Risk calculator"]
)

with tab_overview:
    st.subheader("Exploratory data analysis")
    _fig("eda", "EDA overview")
    st.markdown(
        """
        **What to look for**
        - Month-to-month contracts churn far more than annual contracts.
        - Short tenure + high monthly charges = the classic risk profile.
        - The classes are imbalanced → accuracy is the wrong headline metric;
          we track ROC-AUC, PR-AUC and calibration instead.
        """
    )
    _fig("cv_comparison", "Cross-validated model comparison")

with tab_models:
    st.subheader("Cross-validation (training set)")
    cv_cols = ["val_accuracy", "val_f1", "val_roc_auc", "val_pr_auc", "cv_seconds"]
    st.dataframe(_table(results["cv"], cv_cols), use_container_width=True)

    st.subheader("Held-out test results")
    test_rows = [
        {"model": name, **m} for name, m in sorted(
            results["test"].items(), key=lambda kv: kv[1]["roc_auc"], reverse=True
        )
    ]
    import pandas as pd

    st.dataframe(
        pd.DataFrame(test_rows).set_index("model")[
            ["roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1", "ece"]
        ].round(4),
        use_container_width=True,
    )

    cols = st.columns(2)
    with cols[0]:
        _fig("roc", "ROC curves")
    with cols[1]:
        _fig("pr", "Precision–Recall curves")
    cols = st.columns(2)
    with cols[0]:
        _fig("calibration", "Calibration curves")
    with cols[1]:
        _fig("confusion", "Confusion matrix (best model)")
    _fig("threshold", "Threshold trade-off")
    _fig("learning_curve", "Learning curve")

    hist_key = f"history_{best}"
    if hist_key in results.get("figures", {}):
        st.subheader(f"PyTorch training — `{best}`")
        _fig(hist_key, "Training curves")

with tab_explain:
    st.subheader("Permutation importance (test-set ROC-AUC drop when shuffled)")
    import pandas as pd

    pi = pd.Series(results.get("permutation_importance", {}))
    if len(pi):
        st.bar_chart(pi)
    _fig("permutation_importance", "Permutation importance")
    _fig("shap_tree", "SHAP — tree model (TreeExplainer)")
    _fig("shap_deep", "SHAP — PyTorch MLP (DeepExplainer)")
    for note in results.get("shap_notes", []):
        st.caption(note)
    _fig("clusters", "Unsupervised baseline: K-Means segments")

with tab_score:
    st.subheader("Score a customer")
    st.caption(f"POST {API_URL}/predict — falls back to the local model if the API is down.")

    with st.form("score_form"):
        c1, c2 = st.columns(2)
        with c1:
            tenure = st.number_input("Tenure (months)", 0, 120, 6)
            monthly = st.number_input("Monthly charges ($)", 18.0, 130.0, 85.0)
            total = st.number_input("Total charges ($)", 1.0, 100_000.0, 510.0)
            tickets = st.number_input("Support tickets", 0, 20, 2)
        with c2:
            contract = st.selectbox("Contract", CONTRACTS)
            internet = st.selectbox("Internet service", INTERNET)
            tech = st.selectbox("Tech support", YESNO)
            payment = st.selectbox("Payment method", PAYMENTS)
        submitted = st.form_submit_button("Predict churn risk")

    if submitted:
        payload = {
            "tenure": float(tenure),
            "monthly_charges": float(monthly),
            "total_charges": float(total),
            "num_tickets": float(tickets),
            "contract_type": contract,
            "internet_service": internet,
            "tech_support": tech,
            "payment_method": payment,
        }
        response = _predict_via_api(payload)
        used_api = response is not None
        if response is None:
            response = _predict_local(payload)
        if response is None:
            st.error(
                "No prediction possible — start the API (`make serve`) "
                "or train a model (`make pipeline`)."
            )
        else:
            pred = response["prediction"]
            prob = pred["churn_probability"]
            source = "FastAPI service" if used_api else "local model"
            st.metric("Churn probability", f"{prob:.1%}", help=f"via {source}")
            band = pred["risk_band"]
            color = {"low": "green", "medium": "orange", "high": "red"}[band]
            st.markdown(f"**Risk band: <span style='color:{color}'>{band.upper()}</span>**",
                        unsafe_allow_html=True)
            st.progress(min(prob, 1.0))
            st.caption(
                f"model `{pred['model']}` · "
                "high risk → retention offer / human handoff; "
                "low risk → no action needed."
            )

st.sidebar.header("About")
st.sidebar.markdown(
    """
**Customer Churn Prediction**

Multi-model pipeline:
- scikit-learn (LogReg, RF, GBM)
- XGBoost
- imbalanced-learn (SMOTE)
- PyTorch MLP + TabTransformer

Plus SHAP explainability, MLflow tracking,
FastAPI serving and this dashboard.
"""
)
if results:
    st.sidebar.subheader("Best model")
    st.sidebar.write(f"`{best}`")
    st.sidebar.subheader("Test ROC-AUC")
    st.sidebar.write(f"{results['test'][best]['roc_auc']:.4f}")
    st.sidebar.subheader("API")
    st.sidebar.write(API_URL)
