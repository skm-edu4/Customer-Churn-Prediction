"""Pydantic request/response schemas — the public API contract."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ContractType = Literal["Month-to-month", "One year", "Two year"]
InternetType = Literal["Fiber optic", "DSL", "No internet"]
YesNo = Literal["Yes", "No"]
PaymentType = Literal["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
RiskBand = Literal["low", "medium", "high"]


class CustomerRecord(BaseModel):
    """One customer's raw features — exactly the columns the model was trained on."""

    tenure: float = Field(..., ge=0, le=120, description="Months as a customer")
    monthly_charges: float = Field(..., gt=0, description="Monthly bill amount")
    total_charges: float = Field(..., gt=0, description="Lifetime bill amount")
    num_tickets: float = Field(..., ge=0, description="Support tickets opened")
    contract_type: ContractType
    internet_service: InternetType
    tech_support: YesNo
    payment_method: PaymentType

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "tenure": 2,
                    "monthly_charges": 92.5,
                    "total_charges": 185.0,
                    "num_tickets": 3,
                    "contract_type": "Month-to-month",
                    "internet_service": "Fiber optic",
                    "tech_support": "No",
                    "payment_method": "Electronic check",
                }
            ]
        }
    }


class Prediction(BaseModel):
    churn_probability: float = Field(..., ge=0, le=1)
    risk_band: RiskBand
    model: str


class PredictResponse(BaseModel):
    prediction: Prediction


class BatchRequest(BaseModel):
    customers: list[CustomerRecord] = Field(..., min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    model: str
    count: int
    predictions: list[Prediction]


class ModelInfo(BaseModel):
    best_model: str
    kind: str
    available_models: list[str]
    api_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


def risk_band(probability: float) -> RiskBand:
    """Map a churn probability to a business-facing risk band."""
    if probability < 0.33:
        return "low"
    if probability < 0.66:
        return "medium"
    return "high"
