# 📉 Customer Churn Prediction — Production-Grade ML System

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E.svg)](https://scikit-learn.org/)
[![CI](https://github.com/skm-edu4/Customer-Churn-Prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/skm-edu4/Customer-Churn-Prediction/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Seven models, one pipeline, zero leakage** — a structured ML system that compares classical ML (LogReg, Random Forest, Gradient Boosting), XGBoost, an imbalanced-learn SMOTE variant, and two from-scratch PyTorch models (MLP + TabTransformer), then evaluates, explains, cost-optimises and serves the winner behind a FastAPI endpoint, a Streamlit dashboard and a Docker Compose stack — with ONNX export + latency benchmarks to prove it's fast.

---

## 🧩 Problem → Process → Outcome

**The problem.**
Subscription businesses lose 30–40% of their customers every year, but retention
teams usually find out *after* the cancellation. Support and ops teams had no way
to get a **calibrated, explainable churn risk score** per customer — and naive
accuracy scores actively mislead on imbalanced data (a model that never predicts
"churn" still scores ~64% accuracy here).

**The process.**
A production-style ML system rather than a notebook:

1. **Data** — a deterministic synthetic dataset (seed-reproducible, ground truth
   known) with a drop-in path for the Kaggle *Telco Customer Churn* CSV.
2. **Leakage-free comparison** — 7 model families under one 5-fold CV harness:
   Logistic Regression, Random Forest, Gradient Boosting, XGBoost, an
   imbalanced-learn **SMOTE** pipeline (resampling inside the fold), a PyTorch
   **MLP**, and a from-scratch **TabTransformer** (entity embeddings +
   self-attention).
3. **Honest evaluation** — ROC-AUC, PR-AUC, F1, **calibration (ECE)**, threshold
   trade-off analysis, **cost-sensitive threshold selection** (explicit FP/FN
   prices), learning curves, and a single untouched test set. An opt-in
   **temporal split** with injected concept drift shows what an i.i.d. split
   hides.
4. **Explanation** — SHAP (TreeExplainer + DeepExplainer) and model-agnostic
   permutation importance.
5. **Optimisation** — opt-in **Optuna** (TPE) hyperparameter search per model
   family (`make hpo`), and **ONNX export** of the torch models with a
   parity check (max |Δ| ≈ 1e-07) plus a p50/p95 latency benchmark.
6. **Delivery** — best model behind a **FastAPI** REST contract, an interactive
   **Streamlit** dashboard, MLflow experiment tracking, a Markdown report, a
   **Docker Compose** stack (API + dashboard), 53 tests and GitHub Actions CI.

**The outcome.**
- Best model **Logistic Regression: test ROC-AUC 0.781, PR-AUC 0.658**
  (vs. 0.36 base rate → ~1.8× lift); all 7 models within 5 AUC points —
  measured evidence that well-tuned linear/tree models rival deep nets on
  tabular data.
- **Best-calibrated** model is the PyTorch MLP (ECE 0.049); the from-scratch
  TabTransformer is the strongest deep model (ROC-AUC 0.776).
- **Cost maths beats the default threshold**: with FP cost 1 (wasted offer)
  and FN cost 5 (lost customer), the expected cost is minimised at threshold
  **0.23 → cost 478 vs 725 at 0.50** (34% cheaper).
- **Temporal validation exposes drift**: with injected concept drift (monthly
  churn 0.24 → 0.48), a random split reports ECE 0.046 while a temporal split
  reveals **ECE 0.169** and a cost-optimal threshold of 0.07 — the same
  model, two very different stories.
- Explicit business levers: threshold **0.3 catches 90.6%** of churners
  (673 flags), threshold **0.5 balances at 56% precision / 71% recall** —
  retention spend vs. missed churners becomes a one-line decision.
- Feature insight: `contract_type` alone explains **−0.18 ROC-AUC** when
  shuffled, while `total_charges` and `payment_method` are dead weight.
- **Fast to serve**: ONNX Runtime matches eager PyTorch to **1.2e-07** and
  runs the TabTransformer ~17% faster at batch size 1 (0.290 vs 0.349 ms).
- Fully reproducible: `make setup && make pipeline` in minutes, no API keys,
  no downloads.

## 🎯 What this project does

1. **Trains** 7 model families on a 5,000-row churn dataset (synthetic by default, drop-in compatible with the Kaggle Telco dataset)
2. **Compares** them with 5-fold cross-validation and a held-out test set (optionally a **temporal** split: train on older cohorts, test on newer)
3. **Tunes** on demand with seeded Optuna TPE search (`make hpo`) — off by default so the published baseline stays reproducible
4. **Explains** predictions with SHAP (TreeExplainer + DeepExplainer) and model-agnostic permutation importance
5. **Cost-optimises the decision** — expected-cost curve across thresholds with explicit FP/FN prices (the optimum is 0.23, not 0.50)
6. **Visualises** everything with 16 matplotlib/seaborn figures (ROC, PR, calibration, thresholds, cost curve, learning curves, clusters…)
7. **Exports & benchmarks** both torch models to ONNX with a parity check and p50/p95 latency table
8. **Tracks** experiments in MLflow and emits a Markdown evaluation report
9. **Serves** the best model over REST (FastAPI), a live risk calculator (Streamlit) and a **Docker Compose** stack
10. **Guarantees** quality with 53 tests (unit + end-to-end) and GitHub Actions CI

## 🏗️ Architecture

```mermaid
flowchart TD
    A["configs/default.yaml"] --> B["data.py<br/>synthetic generator or Kaggle Telco CSV<br/>→ cached parquet (drift + split strategy in key)"]
    B --> C["train.py"]
    C -.->|optional --hpo| C0["Optuna TPE search<br/>(seeded, per-family)"]
    C0 -.-> C
    C --> C1["sklearn / XGBoost / SMOTE pipelines<br/>5-fold cross-validation"]
    C --> C2["PyTorch MLP<br/>BatchNorm + early stopping"]
    C --> C3["TabTransformer<br/>entity embeddings + self-attention"]
    C1 --> D["models/bundle.json<br/>registry + best-model selection"]
    C2 --> D
    C3 --> D
    D --> E["evaluate.py<br/>held-out metrics, threshold + cost analysis"]
    E --> F["16 figures<br/>ROC / PR / calibration / cost curve / SHAP / …"]
    E --> G["generate_report.py<br/>reports/REPORT.md"]
    E --> H["MLflow (mlruns/)"]
    D --> K["benchmark.py<br/>ONNX export + parity check<br/>+ p50/p95 latency table"]
    K --> D
    D --> I["FastAPI<br/>/predict /predict/batch /model-info"]
    I --> J["Streamlit dashboard<br/>EDA · comparison · explainability · risk calculator"]
    J -.->|fallback| D
    I --> L["docker compose up<br/>one image, two services (api + dashboard)"]
    J --> L
```

## 📊 Results (seed 42, 5,000 rows, 36.1% churn rate)

**Held-out test set (1,000 rows):**

| Model | ROC-AUC | PR-AUC | F1 | ECE ↓ |
|---|---|---|---|---|
| **logistic_regression** ✅ | **0.781** | **0.658** | **0.627** | 0.097 |
| tabtransformer (PyTorch) | 0.776 | 0.655 | 0.562 | 0.054 |
| gradient_boosting_smote | 0.775 | 0.653 | 0.611 | 0.067 |
| torch_mlp (PyTorch) | 0.773 | 0.648 | 0.555 | 0.049 |
| gradient_boosting | 0.772 | 0.647 | 0.534 | 0.047 |
| xgboost | 0.741 | 0.607 | 0.542 | 0.080 |
| random_forest | 0.730 | 0.586 | 0.559 | 0.087 |

**Business decision layer — cost-sensitive thresholds:**

The threshold isn't a statistic, it's an economic choice. With a false
positive costing **1** (a wasted retention offer) and a false negative
costing **5** (a lost customer), expected cost across thresholds gives:

| Decision | Threshold | Expected cost | Cost/customer |
|---|---|---|---|
| Convention (calibrated 0.50) | 0.50 | 725 | 0.725 |
| **Cost-optimal** | **0.23** | **478** | 0.478 |
| Saving by moving to the optimum | −0.27 | **−247 (−34%)** | −0.247 |

**Temporal validation — what an i.i.d. split hides (drift demo):**

Churn is a temporal event. With `data.temporal_drift: 0.9` injected (monthly
churn climbs from **0.24 → 0.48** across the 12 cohorts) and *the same seed
and model*, only the split changes:

| | random split | temporal split |
|---|---|---|
| Test ROC-AUC | 0.776 | 0.747 |
| Test ECE | 0.046 | **0.169** (3.7× worse) |
| Recall @ 0.50 | 0.51 | 0.32 |
| Cost-optimal threshold | 0.15 (cost 504) | **0.07** (cost 493 vs **1639** at 0.50) |

The random split averages the drift away and the model looks calibrated; the
temporal split shows it calibrated for a 36%-churn world being asked to
predict a 48%-churn one. This is why `split_strategy: temporal` exists —
opt-in so the headline numbers stay comparable, but one YAML line away.

**Inference latency (M-series MacBook, end-to-end `predict_proba`, 200 timed calls):**

| Model | batch 1 p50 | batch 64 p50 |
|---|---|---|
| TabTransformer (ONNX) | **0.290 ms** | 1.427 ms |
| TabTransformer (eager) | 0.349 ms | **0.758 ms** |
| torch_mlp (ONNX) | 1.793 ms | 1.982 ms |
| torch_mlp (eager) | 1.840 ms | 1.924 ms |
| logistic_regression | 1.845 ms | 1.884 ms |
| random_forest (300 trees) | 26.1 ms | 28.2 ms |

ONNX parity vs eager: **max |Δ| = 1.19e-07** for both torch models. ONNX wins
~17% at batch 1 for the transformer — but eager wins at batch 64 here
(single-threaded ORT vs torch's batched kernels at this size). Exporting is
not automatically "faster": you benchmark, you don't assume. Full table
(all 9 predictors × p95 × throughput) in `reports/benchmark.md`.

**Honest takeaways worth discussing in an interview:**

- The synthetic data's ground-truth mechanism *is* a logistic model — so the linear model wins, and the deep models land within **0.5 ROC-AUC points** of it. On tabular data, neural networks rarely beat well-tuned linear/tree models; this repo shows that empirically instead of asserting it.
- The **from-scratch TabTransformer** (entity embeddings + transformer encoder, no pretrained libraries) is the second-best model and the **best-calibrated deep model** (ECE 0.054).
- **Accuracy is the wrong metric here**: at threshold 0.5 the recall/precision trade-off is very different from 0.3 — see the threshold analysis in `reports/REPORT.md`.
- Permutation importance says `contract_type` alone accounts for **−0.18 ROC-AUC** when shuffled; `total_charges` and `payment_method` add nothing once other features are known (correlation, not causation).

## 📸 Screenshots

### 1. Executing the pipeline

**`make train`** — 5-fold CV across all candidates, torch training, model selection:

![make train output](docs/screenshots/exec_01_train.png)

**`make evaluate`** — held-out test metrics + cost analysis + permutation importance + 16 figures:

![make evaluate output](docs/screenshots/exec_02_evaluate.png)

**`make benchmark`** — ONNX export, parity check and p50/p95 latency for all 9 predictors:

![make benchmark output](docs/screenshots/exec_08_benchmark.png)

### 2. Serving the model (FastAPI)

**`curl` against the live API** — health, model info, and two opposite risk profiles:

![API curl execution](docs/screenshots/exec_03_api_curl.png)

**Interactive Swagger docs at `/docs`** — typed schemas, 422 validation, batch endpoint:

![FastAPI Swagger UI](docs/screenshots/exec_04_fastapi_docs.png)

### 3. Streamlit dashboard

**Overview** — dataset, best model, live ROC-AUC:

![Dashboard overview](docs/screenshots/exec_05_dashboard_overview.jpg)

**Model comparison** — all 7 models, ROC/PR/calibration side by side:

![Dashboard model comparison](docs/screenshots/exec_06_dashboard_models.jpg)

**Risk calculator** — score a customer through the API and get a risk band:

![Dashboard risk prediction](docs/screenshots/exec_07_dashboard_prediction.jpg)

### 4. Docker Compose

**`docker compose up -d`** — one image, two services; the dashboard waits for the API's healthcheck, and `models/` is bind-mounted read-only:

![docker compose execution](docs/screenshots/exec_09_docker.png)

### 5. Evaluation & explainability figures

| Figure | |
|---|---|
| EDA overview | ![EDA](docs/screenshots/01_eda.jpg) |
| ROC curves (all models) | ![ROC](docs/screenshots/03_roc_curves.jpg) |
| Calibration | ![Calibration](docs/screenshots/05_calibration.jpg) |
| Confusion matrix (best model) | ![Confusion](docs/screenshots/06_confusion.jpg) |
| Threshold trade-off | ![Threshold](docs/screenshots/07_threshold_curve.jpg) |
| Cost-sensitive threshold (FP/FN economics) | ![Cost curve](docs/screenshots/16_cost_curve.jpg) |
| TabTransformer training curves | ![Training](docs/screenshots/09_history_tabtransformer.jpg) |
| Learning curve (bias–variance) | ![Learning curve](docs/screenshots/10_learning_curve.jpg) |
| Permutation importance | ![Permutation](docs/screenshots/12_permutation_importance.jpg) |
| SHAP — tree model | ![SHAP tree](docs/screenshots/13_shap_tree.jpg) |
| SHAP — PyTorch MLP (DeepExplainer) | ![SHAP deep](docs/screenshots/14_shap_deep_mlp.jpg) |
| Inference latency (eager vs ONNX) | ![Latency](docs/screenshots/17_latency.jpg) |
| Concept drift by cohort (drift demo run) | ![Drift](docs/screenshots/15_drift.jpg) |

> All screenshots were captured from the actual run (seed 42). Regenerate everything with `make pipeline` (the drift figure comes from the temporal-validation demo — set `data.temporal_drift: 0.9`).

## 🧠 Skills & libraries map

| Area | Where |
|---|---|
| **PyTorch** | `src/churn/training/` — custom `nn.Module`s, Dataset/DataLoader, AdamW + cosine LR, early stopping, checkpointing, weighted BCE & focal loss |
| **scikit-learn** | `training/baselines.py` — pipelines, `ColumnTransformer`, `GridSearchCV`-style CV, learning curves, permutation importance |
| **XGBoost** | `training/baselines.py` — tuned `XGBClassifier` in the same CV harness |
| **imbalanced-learn** | `gradient_boosting_smote` — SMOTE *inside* the CV fold (no leakage) |
| **Optuna** | `training/hpo.py` — seeded TPE search per family, `--hpo` opt-in, best params written to `models/hpo_best_params.json` |
| **ONNX Runtime** | `export/onnx_export.py` — dynamic-batch graph export + `Predictor`-compatible ORT sessions; parity asserted in tests |
| **Cost-sensitive ML** | `evaluation/metrics.py: cost_table` — expected-cost curve, economic optimum (0.23 vs 0.50) |
| **Temporal validation** | `data.py: train_test_frames` — drift injection + cohort-ordered split (train old → test new) |
| **matplotlib / seaborn** | `evaluation/plots.py` — 16 publication-style figures |
| **SHAP** | `explain/shap_analysis.py` — TreeExplainer + DeepExplainer + permutation fallback |
| **MLflow** | `tracking.py` — params/metrics/artifacts to a local file store |
| **FastAPI / pydantic** | `serving/` — typed request contract, validation, 422s, batch endpoint, OpenAPI docs |
| **Streamlit** | `dashboard/app.py` — EDA, model comparison, explainability, live scoring |
| **Docker** | `Dockerfile` + `docker-compose.yml` — one image, two services, artefacts bind-mounted ro, health-gated startup |
| **Testing / CI** | `tests/` (53 tests incl. end-to-end), ruff, mypy, GitHub Actions matrix 3.11/3.12 |
| **Packaging** | `pyproject.toml` + uv, src layout, Makefile, typed config (pydantic + YAML) |

## 📁 Project structure

```
Customer-Churn-Prediction/
├── configs/default.yaml          # every knob: seed, model hparams, thresholds, costs, HPO
├── src/churn/
│   ├── config.py                 # pydantic config loader
│   ├── data.py                   # synthetic generator + Telco CSV loader + parquet cache
│   │                             #   + drift injection + random/temporal split
│   ├── features.py               # shared ColumnTransformer + TorchFeatureEncoder
│   ├── models_io.py              # uniform Predictor interface for all 7 models
│   ├── tracking.py               # MLflow wrapper (best-effort, never breaks training)
│   ├── training/
│   │   ├── baselines.py          # LogReg / RF / GBM / XGBoost / SMOTE pipelines
│   │   ├── hpo.py                # Optuna TPE search (seeded, opt-in)
│   │   ├── torch_common.py       # Trainer: early stopping, cosine LR, checkpoints
│   │   ├── torch_mlp.py          # ChurnMLP (BatchNorm + dropout)
│   │   └── tab_transformer.py    # TabTransformer (embeddings + self-attention)
│   ├── evaluation/               # metrics (ROC/PR/ECE/thresholds/cost) + plots
│   ├── export/onnx_export.py     # ONNX export + ORT predictors + parity check
│   ├── explain/shap_analysis.py  # SHAP + permutation importance
│   └── serving/                  # FastAPI app + pydantic schemas
├── scripts/                      # train · evaluate · benchmark · generate_report
├── dashboard/app.py              # Streamlit
├── tests/                        # 53 tests incl. end-to-end + ONNX/HPO units
├── .github/workflows/ci.yml      # ruff → mypy → pytest on 3.11 & 3.12
├── Dockerfile                    # one image → api + dashboard services
├── docker-compose.yml
├── Makefile
└── reports/                      # generated: figures + REPORT.md + benchmark.md
```

## 🚀 Getting started

Requires [uv](https://docs.astral.sh/uv/) (or plain pip).

```bash
git clone https://github.com/skm-edu4/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction

make setup                 # uv sync — creates .venv, installs everything

make pipeline              # train → evaluate → benchmark → report  (models/, reports/)
make serve                 # FastAPI  → http://localhost:8000/docs
make dashboard             # Streamlit → http://localhost:8501
```

Or the same two services in containers (after `make pipeline` has produced
`models/`, which compose bind-mounts read-only):

```bash
docker compose up --build   # API → :8000, dashboard → :8501 (health-gated startup)
```

No API keys, no downloads — the dataset is generated deterministically from `seed`.

<details>
<summary>pip alternative (no uv)</summary>

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install pytest ruff mypy httpx types-PyYAML pandas-stubs   # dev tools
python scripts/train.py && python scripts/evaluate.py && python scripts/generate_report.py
```
</details>

### Use the real Telco dataset

```bash
# from https://www.kaggle.com/datasets/blastchar/telco-customer-churn
mv telco.csv data/raw/telco_customer_churn.csv
# set `data.source: telco` in configs/default.yaml
make pipeline
```

### Everyday commands

| Command | What it does |
|---|---|
| `make train` | CV + fit all 7 models, select best, write `models/bundle.json` |
| `make hpo` | Same, but first runs the seeded Optuna search (`train.py --hpo`) |
| `make evaluate` | Held-out metrics + cost analysis + 16 figures + SHAP |
| `make benchmark` | ONNX export + parity check + latency table (`reports/benchmark.md`) |
| `make report` | Build `reports/REPORT.md` |
| `make test` | pytest (unit + end-to-end) |
| `make lint typecheck` | ruff + mypy |
| `make docker-up` / `make docker-down` | API + dashboard in containers |
| `make clean` | Remove generated artefacts |

## 🛡️ Design decisions (the interview-defence bits)

**Why a `bundle.json` + one `Predictor` interface?**
Seven models with three different loading mechanisms (joblib pipeline, torch + preprocessor, torch + encoder) are useless to a serving layer that must load "the best one". `models_io.py` normalises all of them behind `predict_proba(DataFrame)` — the API and dashboard don't know (or care) what's inside.

**Why is preprocessing inside the pipeline?**
Every sklearn candidate wraps imputation/scaling/one-hot in the `Pipeline`, so CV folds never see statistics from validation rows. The torch models use a sub-split of the training set for early stopping, and encoders are fit on the sub-train fold only.

**Why SMOTE inside an `imblearn.Pipeline`?**
Resampling before the split would leak synthetic points across folds. imblearn's pipeline applies SMOTE only to training folds.

**Why report ECE (calibration) and PR-AUC?**
Churn is imbalanced (36%) and the output is used as a *risk score*, not just a label — a confidently-wrong 0.9 costs more than an uncertain 0.5. Accuracy alone hides both problems.

**Why a from-scratch TabTransformer instead of `torchtabular`?**
To be able to explain every parameter: entity-embedding tables sized from the data, a `[CLS]` token, positional embeddings, `TransformerEncoder` blocks, then a continuous-feature head.

**Why a cost curve instead of picking threshold 0.50 / max-F1?**
The threshold encodes business economics, not statistics. A false positive is a wasted retention offer (cost 1); a false negative is a lost customer (cost 5). Minimising `fp(t)·1 + fn(t)·5` over a 0.01 threshold grid puts the optimum at **0.23** — expected cost **478 vs 725 at 0.50** (34% cheaper). The report plots the full FP/FN breakdown so the trade-off is visible, and the direction is provable: as FN cost grows, the optimum can only move left (fp is non-increasing, fn non-decreasing in t).

**Why is the temporal split opt-in instead of the default?**
Because the headline table has to stay comparable with the published run — switching the split re-baselines every number. So `split_strategy: random` stays the default and `temporal` is one YAML line away. The honest engineering detail: `observation_month` is drawn from a **separate RNG stream**, so with `temporal_drift: 0.0` every other column is byte-identical to the original generator (a test asserts this). The month is metadata — dropped from `X` by `split_features_target`, and both encoders select columns by explicit name lists, so it can never leak as a feature.

**Why is Optuna opt-in (`make hpo`) rather than always-on?**
TPE with a seeded sampler makes HPO itself reproducible — but it still changes chosen hyperparameters, which changes every number in the README. Default config keeps `hpo.enabled: false` (a test asserts it); turn it on when you *want* to beat the baseline. Answer line: *"HPO is a deliberate experiment, not a background process."*

**ONNX export — what actually broke (good debugging story):**
Torch 2.14 defaults to the dynamo exporter; for the two-input TabTransformer it silently baked a static batch dim (`[1, k]` traced — batch 1 worked, batch 400 crashed in ONNX Runtime). Pinning the legacy exporter (`dynamo=False`) fixed that — which then failed on `aten::_transformer_encoder_layer_fwd`: bisecting grad/no-grad × output-name variants showed `torch.no_grad()` was the trigger (without grad mode the tracer takes TransformerEncoder's fused fast-path, which has no ONNX symbolic). Removed the wrapper, pinned and documented both constraints in `export/onnx_export.py`. Verification on every run: max |Δ| eager-vs-ORT = **1.19e-07**, asserted in unit tests and reported by `make benchmark`. And the benchmark keeps everyone honest: ONNX wins ~17% at batch 1 but *loses* at batch 64 here — export is not automatically faster.

**Docker: one image, two commands — and no baked-in models.**
`Dockerfile` ships code only; `docker-compose.yml` bind-mounts `./models` and `./reports` **read-only**. Retraining = restart, not rebuild; the API container can't be mutated at runtime; the image is shareable without artefacts. Dependencies layer-cache (`uv sync --no-install-project` first), runtime uses plain venv binaries (no package manager at runtime), non-root user, healthchecks on `/health` and Streamlit's `/_stcore/health`, and the dashboard waits on `service_healthy` so it never races the API. The private `INTERVIEW_NOTES.md` is excluded by both `.gitignore` and `.dockerignore` — it ships in neither git nor the image.

**The macOS dual-libomp workaround (read this if you port the repo):**
`xgboost` links `@rpath/libomp.dylib` while PyTorch bundles its *own* `libomp.dylib`. In one process, two OpenMP runtimes deadlock (or segfault) inside their barriers. `churn/__init__.py` pins `OMP_NUM_THREADS=1` before either library loads — negligible cost at this data size, and the reason the pipeline is stable on macOS. See the comment there for details.

**Why did the linear model win?**
See the honest results discussion above — and be ready for the follow-up: *"what would you do with real data?"* → richer features (usage aggregates, support-sentiment), and then the parts this repo already prototypes: time-aware splits (`split_strategy: temporal`), drift monitoring, and cost-based threshold selection calibrated to retention economics.

## ✅ Testing & CI

```bash
make test        # 53 tests, ~2 min
```

- `test_data.py` — generator determinism, schema, parquet caching (incl. drift cache-key), temporal split ordering + Telco fallback
- `test_features.py` — leakage-free transforms, unseen-category handling, picklability
- `test_metrics.py` — ROC/PR/ECE math, threshold monotonicity, cost math + the FP/FN comparative-statics property
- `test_hpo.py` — Optuna search returns usable, reproducible params; overrides don't touch other families
- `test_onnx_export.py` — eager ↔ ONNX Runtime parity (< 1e-3) for both torch models, batch-independence
- `test_torch_model.py` — forward shapes, learning a separable pattern, seeded determinism, focal loss
- `test_api.py` — FastAPI contract: health, model-info, 422 validation, batch, 503 without a model
- `test_pipeline_integration.py` — runs `train.py` → `evaluate.py` → `benchmark.py` → `generate_report.py` end-to-end in a temp dir (cost + benchmark artefacts asserted)

CI (`.github/workflows/ci.yml`) runs ruff → ruff-format → mypy → pytest on **Python 3.11 and 3.12**.

## 🔮 Roadmap

- [x] Time-based split (churn is a temporal event, not i.i.d.) — `data.split_strategy: temporal` + `data.temporal_drift` injection
- [x] Optuna hyperparameter search replacing the manual grids — `training/hpo.py`, opt-in via `make hpo`
- [x] ONNX export for the torch models + latency benchmark — `make benchmark`, parity asserted in tests
- [x] Dockerfile + docker-compose for API + dashboard — one image, two services, health-gated
- [x] Cost-sensitive thresholding (explicit false-positive / false-negative prices) — `evaluation/metrics.py: cost_table`
- [ ] DVC for dataset/versioned experiments *(skipped: the dataset is seed-generated from config — DVC would version a file that's already reproducible; revisit if a curated/real dataset with manual annotations lands)*

## 📄 License

MIT — see [LICENSE](LICENSE).

## 🙏 Acknowledgments

Built with [scikit-learn](https://scikit-learn.org/), [PyTorch](https://pytorch.org/), [XGBoost](https://xgboost.readthedocs.io/), [imbalanced-learn](https://imbalanced-learn.org/), [Optuna](https://optuna.org/), [ONNX Runtime](https://onnxruntime.ai/), [SHAP](https://shap.readthedocs.io/), [MLflow](https://mlflow.org/), [FastAPI](https://fastapi.tiangolo.com/), [Streamlit](https://streamlit.io/) and [Docker](https://www.docker.com/).

---

⭐ If this helped you learn, consider starring the repo!
