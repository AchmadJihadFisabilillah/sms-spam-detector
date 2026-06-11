from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "SMS Spam Detection API")
    app_version: str = os.getenv("APP_VERSION", "1.0.0")
    data_path: Path = Path(os.getenv("DATA_PATH", "data/spam.csv"))
    artifact_dir: Path = Path(os.getenv("ARTIFACT_DIR", "artifacts"))
    cors_origins: tuple[str, ...] = tuple(
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173",
        ).split(",")
        if origin.strip()
    )

    @property
    def metrics_path(self) -> Path:
        return self.artifact_dir / "metrics.json"

    @property
    def predictions_path(self) -> Path:
        return self.artifact_dir / "test_predictions.csv"

    @property
    def naive_bayes_model_path(self) -> Path:
        return self.artifact_dir / "naive_bayes_model.joblib"

    @property
    def logistic_regression_model_path(self) -> Path:
        return self.artifact_dir / "logistic_regression_model.joblib"


settings = Settings()
