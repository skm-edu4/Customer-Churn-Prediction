"""Dataset creation and loading.

Two sources are supported:

* ``synthetic`` — a realistic churn dataset generated in code (fully
  reproducible from a seed, no downloads needed).
* ``telco``     — the public Kaggle *Telco Customer Churn* CSV, mapped onto
  the same schema so the rest of the pipeline is source-agnostic.

Every synthetic row also carries an ``observation_month`` (1–12) — the cohort
the customer was observed in. It is metadata, never a model feature: it exists
so the temporal (non-i.i.d.) train/test split and the concept-drift analysis
have something to order rows by.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from churn.config import Config

CATEGORICAL_COLUMNS = [
    "contract_type",
    "internet_service",
    "tech_support",
    "payment_method",
]
NUMERIC_COLUMNS = ["tenure", "monthly_charges", "total_charges", "num_tickets"]
TARGET_COLUMN = "churned"
TIME_COLUMN = "observation_month"

CONTRACT_MAP = {
    "Month-to-month": "Month-to-month",
    "One year": "One year",
    "Two year": "Two year",
}
INTERNET_MAP = {
    "Fiber optic": "Fiber optic",
    "DSL": "DSL",
    "No": "No internet",
}
PAYMENT_MAP = {
    "Electronic check": "Electronic check",
    "Mailed check": "Mailed check",
    "Bank transfer (automatic)": "Bank transfer",
    "Credit card (automatic)": "Credit card",
}


def generate_churn_dataset(
    n_customers: int = 5000, seed: int = 42, *, temporal_drift: float = 0.0
) -> pd.DataFrame:
    """Generate a synthetic but realistic churn dataset.

    The hidden churn mechanism encodes the classic drivers (month-to-month
    contracts, fiber optic, no tech support, high charges, low tenure) so the
    models have genuine structure to discover.

    ``temporal_drift`` optionally adds a linear trend to the churn log-odds
    over ``observation_month`` (simulating e.g. a price increase that makes
    recent cohorts churn more). The month is drawn from a **separate RNG
    stream**, so with ``temporal_drift=0.0`` every other column is
    byte-identical to the original generator.
    """
    rng = np.random.default_rng(seed)

    tenure = np.clip(rng.normal(32, 22, n_customers), 1, 72).round()

    contract_type = rng.choice(
        ["Month-to-month", "One year", "Two year"], size=n_customers, p=[0.55, 0.25, 0.20]
    )
    internet_service = rng.choice(
        ["Fiber optic", "DSL", "No internet"], size=n_customers, p=[0.45, 0.35, 0.20]
    )
    tech_support = rng.choice(["Yes", "No"], size=n_customers, p=[0.40, 0.60])
    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        size=n_customers,
        p=[0.35, 0.25, 0.20, 0.20],
    )

    base = {"Fiber optic": 85.0, "DSL": 60.0, "No internet": 25.0}
    monthly_charges = np.array([base[s] for s in internet_service]) + rng.normal(0, 12, n_customers)
    monthly_charges = np.clip(monthly_charges, 18, 120).round(2)

    total_charges = (monthly_charges * tenure * rng.normal(1.0, 0.05, n_customers)).round(2)

    num_tickets = rng.poisson(1.2, n_customers) + rng.poisson(
        np.where(monthly_charges > 80, 0.6, 0.1), n_customers
    )

    # Cohort month: independent stream so the drift-free dataset is unchanged.
    month_rng = np.random.default_rng(seed + 991)
    observation_month = month_rng.integers(1, 13, size=n_customers)

    logit = (
        -1.40
        + 1.50 * (contract_type == "Month-to-month")
        - 0.90 * (contract_type == "Two year")
        + 0.80 * (internet_service == "Fiber optic")
        - 0.70 * (tech_support == "Yes")
        + 0.30 * num_tickets
        + 0.012 * (monthly_charges - 65)
        - 0.018 * tenure
        + rng.normal(0, 0.8, n_customers)
    )
    if temporal_drift != 0.0:
        # Later cohorts churn more: ±temporal_drift log-odds from month 1 → 12.
        logit = logit + temporal_drift * (observation_month - 6.5) / 5.5
    churned = (rng.random(n_customers) < 1 / (1 + np.exp(-logit))).astype(int)

    df = pd.DataFrame(
        {
            "tenure": tenure,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "num_tickets": num_tickets.astype(float),
            "contract_type": contract_type,
            "internet_service": internet_service,
            "tech_support": tech_support,
            "payment_method": payment_method,
            TIME_COLUMN: observation_month.astype(int),
            "churned": churned,
        }
    )

    # Real data is messy: inject some missing values
    df.loc[rng.random(n_customers) < 0.03, "total_charges"] = np.nan
    df.loc[rng.random(n_customers) < 0.02, "num_tickets"] = np.nan
    return df


def load_telco_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load the Kaggle Telco churn CSV and map it to the project schema."""
    df = pd.read_csv(csv_path)
    df = df.replace(" ", np.nan)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    mapped = pd.DataFrame(
        {
            "tenure": df["tenure"].astype(float),
            "monthly_charges": df["MonthlyCharges"].astype(float),
            "total_charges": df["TotalCharges"].astype(float),
            "num_tickets": np.nan,  # not present in the Telco dataset
            "contract_type": df["Contract"].map(CONTRACT_MAP),
            "internet_service": df["InternetService"].map(INTERNET_MAP),
            "tech_support": df["TechSupport"].map(
                {"Yes": "Yes", "No": "No", "No internet service": "No"}
            ),
            "payment_method": df["PaymentMethod"].map(PAYMENT_MAP),
            "churned": (df["Churn"] == "Yes").astype(int),
        }
    )
    return mapped


def load_dataset(cfg: Config) -> pd.DataFrame:
    """Load the dataset described by the config (cached to parquet)."""
    cache_path = cfg.processed_data_path
    source_tag = cfg.data.source
    n_tag = cfg.data.n_customers if source_tag == "synthetic" else "telco"
    # The drift setting changes labels, so it must be part of the cache key.
    expected = f"{source_tag}-{cfg.seed}-{n_tag}-d{cfg.data.temporal_drift}"

    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        meta_path = cache_path.with_suffix(".meta")
        if meta_path.exists() and meta_path.read_text().strip() == expected:
            return cached

    if cfg.data.source == "synthetic":
        df = generate_churn_dataset(
            cfg.data.n_customers, cfg.seed, temporal_drift=cfg.data.temporal_drift
        )
    else:
        csv_path = cfg.project_root / cfg.data.telco_csv
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Telco CSV not found at {csv_path}. Download it from "
                "https://www.kaggle.com/datasets/blastchar/telco-customer-churn "
                "and place it there, or set data.source: synthetic."
            )
        df = load_telco_dataset(csv_path)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    cache_path.with_suffix(".meta").write_text(expected)
    return df


def split_features_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) with the target and time metadata columns separated.

    ``observation_month`` never reaches a model: it would be pure leakage
    (it encodes *when* the row was observed, not *who* the customer is).
    """
    drop = [c for c in (TARGET_COLUMN, TIME_COLUMN) if c in df.columns]
    return df.drop(columns=drop), df[TARGET_COLUMN]


def train_test_frames(
    df: pd.DataFrame, cfg: Config
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Deterministic train/test split under the configured strategy.

    Both ``train.py`` and ``evaluate.py`` call this with the same config so
    the test set is identical — evaluate.py never needs the model-training
    code.

    * ``random``  — stratified shuffle split (the i.i.d. assumption churn data
      rarely satisfies, but the standard baseline).
    * ``temporal`` — train on the earliest cohorts, test on the latest ones.
      Rows keep their original relative order inside each cohort. Falls back
      to the random strategy (with a warning) when the dataset has no time
      column (e.g. the Telco CSV).
    """
    X, y = split_features_target(df)

    if cfg.data.split_strategy == "temporal":
        if TIME_COLUMN in df.columns:
            n_test = int(round(len(df) * cfg.data.test_size))
            order = df[TIME_COLUMN].to_numpy()
            # Stable sort: ties keep their original row order (deterministic).
            sorted_idx = np.argsort(order, kind="stable")
            test_idx = sorted_idx[-n_test:]
            train_idx = sorted_idx[:-n_test]
            return (
                X.iloc[train_idx],
                X.iloc[test_idx],
                y.iloc[train_idx],
                y.iloc[test_idx],
            )
        print("[data] temporal split requested but dataset has no time column — using random.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.data.test_size, stratify=y, random_state=cfg.seed
    )
    return X_train, X_test, y_train, y_test
