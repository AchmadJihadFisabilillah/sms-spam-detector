from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    test_size: float = Field(default=0.2, ge=0.1, le=0.4)
    random_state: int = Field(default=42, ge=0)
    max_features: int = Field(default=8000, ge=1000, le=50000)


class PredictRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000)
    threshold: float = Field(default=0.5, ge=0.05, le=0.95)


class ModelPrediction(BaseModel):
    model_name: str
    spam_probability: float
    ham_probability: float
    prediction_label: str
    prediction_value: int


class PredictResponse(BaseModel):
    raw_message: str
    cleaned_message: str
    threshold: float
    best_model: str | None
    final_prediction_label: str
    final_spam_probability: float
    risk_level: str
    predictions: list[ModelPrediction]
    translated_message: str | None = None


class DatasetStatus(BaseModel):
    dataset_exists: bool
    data_path: str
    expected_columns: list[str]
    rows: int | None = None
    label_distribution: dict[str, int] | None = None
    message: str


class TrainResponse(BaseModel):
    status: str
    message: str
    metrics: dict[str, Any]


class ApiMessage(BaseModel):
    message: str
