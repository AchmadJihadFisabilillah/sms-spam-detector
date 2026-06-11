from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .config import settings
from .ml_service import (
    DatasetError,
    ModelNotTrainedError,
    dataset_status,
    load_metrics,
    predict_message,
    train_models,
)
from .schemas import ApiMessage, DatasetStatus, PredictRequest, PredictResponse, TrainRequest, TrainResponse

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="API untuk project akhir Deteksi SMS Spam menggunakan Multinomial Naive Bayes, Logistic Regression, dan McNemar Test.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=ApiMessage)
def health_check() -> dict[str, str]:
    return {"message": "API aktif dan siap digunakan."}


@app.get("/api/project")
def project_info() -> dict:
    return {
        "title": "Deteksi SMS Spam Menggunakan Naive Bayes dan Logistic Regression",
        "dataset": "SMS Spam Collection Dataset dari Kaggle",
        "algorithms": ["Multinomial Naive Bayes", "Logistic Regression"],
        "comparison_method": "McNemar Test + Weighted Model Ranking",
        "target": {"ham": 0, "spam": 1},
        "features": ["TF-IDF unigram/bigram", "engineered text tokens", "keyword indicators", "length bucket"],
    }


@app.get("/api/dataset/status", response_model=DatasetStatus)
def get_dataset_status() -> dict:
    return dataset_status()


@app.post("/api/dataset/upload", response_model=DatasetStatus)
def upload_dataset(file: UploadFile = File(...)) -> dict:
    filename = file.filename or ""
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="File harus berformat .csv")

    settings.data_path.parent.mkdir(parents=True, exist_ok=True)
    with settings.data_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    status = dataset_status()
    if not status["dataset_exists"] or status["rows"] is None:
        raise HTTPException(status_code=400, detail=status["message"])
    return status


@app.post("/api/train", response_model=TrainResponse)
def train(request: TrainRequest) -> dict:
    try:
        metrics = train_models(
            test_size=request.test_size,
            random_state=request.random_state,
            max_features=request.max_features,
        )
        return {
            "status": "success",
            "message": "Training selesai. Model, metrik, dan hasil prediksi berhasil disimpan.",
            "metrics": metrics,
        }
    except DatasetError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Training gagal: {exc}") from exc


@app.get("/api/metrics")
def metrics() -> dict:
    try:
        return load_metrics()
    except ModelNotTrainedError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> dict:
    try:
        return predict_message(request.message, threshold=request.threshold)
    except ModelNotTrainedError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/batch-predict")
def batch_predict(file: UploadFile = File(...), threshold: float = 0.5) -> dict:
    filename = file.filename or ""
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="File batch harus CSV.")

    try:
        df = pd.read_csv(file.file)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"CSV gagal dibaca: {exc}") from exc

    message_column = None
    for candidate in ["message", "text", "sms", "v2"]:
        if candidate in df.columns:
            message_column = candidate
            break
    if message_column is None:
        raise HTTPException(status_code=400, detail="CSV harus memiliki kolom message, text, sms, atau v2.")

    try:
        rows = []
        for raw_message in df[message_column].fillna("").astype(str):
            result = predict_message(raw_message, threshold=threshold)
            rows.append(
                {
                    "message": raw_message,
                    "final_prediction": result["final_prediction_label"],
                    "final_spam_probability": result["final_spam_probability"],
                    "risk_level": result["risk_level"],
                }
            )
        output = pd.DataFrame(rows)
        output_path = settings.artifact_dir / "batch_predictions.csv"
        settings.artifact_dir.mkdir(parents=True, exist_ok=True)
        output.to_csv(output_path, index=False)
        return {
            "status": "success",
            "rows": len(output),
            "download_url": "/api/download/batch-predictions",
            "preview": output.head(10).to_dict(orient="records"),
        }
    except ModelNotTrainedError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/download/predictions")
def download_predictions() -> FileResponse:
    if not settings.predictions_path.exists():
        raise HTTPException(status_code=404, detail="File hasil prediksi belum tersedia. Jalankan training terlebih dahulu.")
    return FileResponse(settings.predictions_path, filename="test_predictions.csv")


@app.get("/api/download/metrics")
def download_metrics() -> FileResponse:
    if not settings.metrics_path.exists():
        raise HTTPException(status_code=404, detail="File metrik belum tersedia. Jalankan training terlebih dahulu.")
    return FileResponse(settings.metrics_path, filename="metrics.json")


@app.get("/api/download/batch-predictions")
def download_batch_predictions() -> FileResponse:
    path = settings.artifact_dir / "batch_predictions.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="File batch prediction belum tersedia.")
    return FileResponse(path, filename="batch_predictions.csv")
