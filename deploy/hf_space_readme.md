---
title: Customer Churn — Model Dashboard
emoji: 📉
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# 📉 Customer Churn — Live Demo

The full stack from **[skm-edu4/Customer-Churn-Prediction](https://github.com/skm-edu4/Customer-Churn-Prediction)**
running in one container:

| Surface | Where |
|---|---|
| **Streamlit dashboard** (this page's app) | the Space URL → EDA · model comparison · explainability · **live risk calculator** |
| **FastAPI** (inside the container, loopback only) | `/predict`, `/predict/batch`, `/model-info` — exercised by the dashboard |

## Try it

Open the Space and use the **Risk calculator** tab — pick customer attributes
(contract, tenure, charges…) and get a churn probability + risk band from the
best model (logistic regression, test ROC-AUC 0.781), served by the API
through the dashboard.

## How it works

- The image **bakes the trained models in at build time** — the pipeline
  (train → evaluate → benchmark → report) runs during `docker build` from a
  deterministic seed, so this demo is reproducible from code alone.
- Deploys automatically: every push to `main` on GitHub triggers
  `.github/workflows/deploy-hf.yml`, which pushes this repo to the Space and
  lets the platform build the Docker image.
- CPU-only dependencies (the CUDA wheels are excluded at build time), so cold
  starts are fast and the image stays small.

## Run it locally

```bash
git clone https://github.com/skm-edu4/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
docker build -t customer-churn .
docker run -p 7860:7860 customer-churn   # dashboard on :7860
```

MIT licensed. Built with PyTorch, scikit-learn, XGBoost, SHAP, FastAPI and Streamlit.
