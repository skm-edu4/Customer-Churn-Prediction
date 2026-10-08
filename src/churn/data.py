"""Dataset creation and loading.

Two sources are supported:

* ``synthetic`` — a realistic churn dataset generated in code (fully
  reproducible from a seed, no downloads needed).
* ``telco``     — the public Kaggle *Telco Customer Churn* CSV, mapped onto
  the same schema so the rest of the pipeline is source-agnostic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from churn.config import Config

CATEGORICAL_COLUMNS = [
    "contract_type",
    "internet_service",
    "tech_support",
    "payment_method",
]
NUMERIC_COLUMNS = ["tenure", "monthly_charges", "total_charges", "num_tickets"]
TARGET_COLUMN = "churned"

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


def generate_churn_dataset(n_customers: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic but realistic churn dataset.

    The hidden churn mechanism encodes the classic drivers (month-to-month
    contracts, fiber optic, no tech support, high charges, low tenure) so the
    models have genuine structure to discover.
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
    expected = f"{source_tag}-{cfg.seed}-{n_tag}"

    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        meta_path = cache_path.with_suffix(".meta")
        if meta_path.exists() and meta_path.read_text().strip() == expected:
            return cached

    if cfg.data.source == "synthetic":
        df = generate_churn_dataset(cfg.data.n_customers, cfg.seed)
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
    """Return (X, y) with the target column separated."""
    return df.drop(columns=[TARGET_COLUMN]), df[TARGET_COLUMN]
