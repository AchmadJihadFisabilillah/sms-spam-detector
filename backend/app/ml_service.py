from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from statsmodels.stats.contingency_tables import mcnemar

from .config import settings
from .text_processing import clean_text, prepare_text_for_model, risk_level

LABEL_MAP = {"ham": 0, "spam": 1}
REVERSE_LABEL_MAP = {0: "ham", 1: "spam"}


class DatasetError(RuntimeError):
    pass


class ModelNotTrainedError(RuntimeError):
    pass


def _read_csv_flexible(path: Path) -> pd.DataFrame:
    errors: list[str] = []
    for encoding in ("latin-1", "utf-8", "cp1252"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception as exc:  # pragma: no cover - kept for robust local use
            errors.append(f"{encoding}: {exc}")
    raise DatasetError("Dataset gagal dibaca. Detail: " + " | ".join(errors))


def load_dataset(path: Path | None = None) -> pd.DataFrame:
    path = path or settings.data_path
    if not path.exists():
        raise DatasetError(
            f"Dataset tidak ditemukan di {path}. Upload spam.csv dari Kaggle atau jalankan scripts/download_kaggle_dataset.py."
        )

    df = _read_csv_flexible(path)

    if {"v1", "v2"}.issubset(df.columns):
        df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "message"})
    elif {"label", "message"}.issubset(df.columns):
        df = df[["label", "message"]].copy()
    elif {"label", "text"}.issubset(df.columns):
        df = df[["label", "text"]].rename(columns={"text": "message"})
    else:
        raise DatasetError(
            "Format kolom tidak sesuai. Dataset harus punya kolom v1/v2, label/message, atau label/text."
        )

    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df["message"] = df["message"].fillna("").astype(str)
    df = df[df["label"].isin(LABEL_MAP)].drop_duplicates(subset=["label", "message"]).reset_index(drop=True)

    if df.empty:
        raise DatasetError("Dataset kosong setelah dibersihkan.")
    if df["label"].nunique() < 2:
        raise DatasetError("Dataset harus memiliki dua kelas: ham dan spam.")

    df["label_encoded"] = df["label"].map(LABEL_MAP)
    df["clean_message"] = df["message"].apply(clean_text)
    df["model_message"] = df["message"].apply(prepare_text_for_model)
    df["message_length"] = df["clean_message"].apply(len)
    df["word_count"] = df["clean_message"].apply(lambda text: len(text.split()))
    return df


def dataset_status() -> dict[str, Any]:
    path = settings.data_path
    if not path.exists():
        return {
            "dataset_exists": False,
            "data_path": str(path),
            "expected_columns": ["v1", "v2"],
            "rows": None,
            "label_distribution": None,
            "message": "Dataset belum ada. Upload spam.csv dari Kaggle melalui web app atau letakkan di backend/data/spam.csv.",
        }
    try:
        df = load_dataset(path)
        return {
            "dataset_exists": True,
            "data_path": str(path),
            "expected_columns": ["v1", "v2"],
            "rows": int(len(df)),
            "label_distribution": {k: int(v) for k, v in df["label"].value_counts().to_dict().items()},
            "message": "Dataset ditemukan dan format valid.",
        }
    except Exception as exc:
        return {
            "dataset_exists": True,
            "data_path": str(path),
            "expected_columns": ["v1", "v2"],
            "rows": None,
            "label_distribution": None,
            "message": f"Dataset ditemukan tetapi format belum valid: {exc}",
        }


def _build_models(max_features: int) -> dict[str, Pipeline]:
    return {
        "naive_bayes": Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        stop_words="english",
                        max_features=max_features,
                        ngram_range=(1, 2),
                        min_df=1,
                        sublinear_tf=False,
                    ),
                ),
                ("model", MultinomialNB(alpha=0.5)),
            ]
        ),
        "logistic_regression": Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        stop_words="english",
                        max_features=max_features,
                        ngram_range=(1, 2),
                        min_df=1,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "model",
                    LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear"),
                ),
            ]
        ),
    }


def _safe_float(value: float | np.floating) -> float:
    value = float(value)
    if np.isnan(value) or np.isinf(value):
        return 0.0
    return value


def _evaluate_model(name: str, y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report = classification_report(y_true, y_pred, target_names=["ham", "spam"], output_dict=True, zero_division=0)
    return {
        "model_key": name,
        "model_name": "Multinomial Naive Bayes" if name == "naive_bayes" else "Logistic Regression",
        "accuracy": _safe_float(accuracy_score(y_true, y_pred)),
        "precision": _safe_float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": _safe_float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": _safe_float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": _safe_float(roc_auc_score(y_true, y_proba)),
        "pr_auc": _safe_float(average_precision_score(y_true, y_proba)),
        "log_loss": _safe_float(log_loss(y_true, y_proba, labels=[0, 1])),
        "brier_score": _safe_float(brier_score_loss(y_true, y_proba)),
        "confusion_matrix": cm.astype(int).tolist(),
        "classification_report": report,
    }


def _rank_models(model_metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = []
    for metric in model_metrics:
        calibration_score = max(0.0, 1.0 - metric["brier_score"])
        final_score = (
            0.30 * metric["roc_auc"]
            + 0.25 * metric["f1_score"]
            + 0.20 * metric["precision"]
            + 0.15 * metric["recall"]
            + 0.10 * calibration_score
        )
        row = dict(metric)
        row["calibration_score"] = _safe_float(calibration_score)
        row["final_score"] = _safe_float(final_score)
        ranked.append(row)
    return sorted(ranked, key=lambda item: item["final_score"], reverse=True)


def _mcnemar_result(y_true: pd.Series, nb_pred: np.ndarray, lr_pred: np.ndarray) -> dict[str, Any]:
    y_true_np = y_true.to_numpy()
    nb_correct = nb_pred == y_true_np
    lr_correct = lr_pred == y_true_np

    both_correct = int(np.sum(nb_correct & lr_correct))
    nb_only = int(np.sum(nb_correct & ~lr_correct))
    lr_only = int(np.sum(~nb_correct & lr_correct))
    both_wrong = int(np.sum(~nb_correct & ~lr_correct))
    table = [[both_correct, nb_only], [lr_only, both_wrong]]

    exact = (nb_only + lr_only) < 25
    result = mcnemar(table, exact=exact, correction=not exact)
    p_value = _safe_float(result.pvalue)

    return {
        "table": table,
        "statistic": _safe_float(result.statistic if result.statistic is not None else 0.0),
        "p_value": p_value,
        "exact_test": bool(exact),
        "alpha": 0.05,
        "is_significant": bool(p_value < 0.05),
        "interpretation": (
            "Perbedaan performa kedua model signifikan secara statistik."
            if p_value < 0.05
            else "Perbedaan performa kedua model tidak signifikan secara statistik."
        ),
    }


def train_models(test_size: float = 0.2, random_state: int = 42, max_features: int = 8000) -> dict[str, Any]:
    settings.artifact_dir.mkdir(parents=True, exist_ok=True)
    df = load_dataset()

    X = df["model_message"]
    y = df["label_encoded"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    models = _build_models(max_features=max_features)
    predictions: dict[str, np.ndarray] = {}
    probabilities: dict[str, np.ndarray] = {}
    model_metrics: list[dict[str, Any]] = []

    for model_key, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        predictions[model_key] = y_pred
        probabilities[model_key] = y_proba
        model_metrics.append(_evaluate_model(model_key, y_test, y_pred, y_proba))

    joblib.dump(models["naive_bayes"], settings.naive_bayes_model_path)
    joblib.dump(models["logistic_regression"], settings.logistic_regression_model_path)

    ranked_models = _rank_models(model_metrics)
    mcnemar_stats = _mcnemar_result(y_test, predictions["naive_bayes"], predictions["logistic_regression"])

    predictions_df = pd.DataFrame(
        {
            "message": df.loc[X_test.index, "message"].to_numpy(),
            "clean_message": df.loc[X_test.index, "clean_message"].to_numpy(),
            "actual_label": y_test.map(REVERSE_LABEL_MAP).to_numpy(),
            "actual_value": y_test.to_numpy(),
            "naive_bayes_prediction": [REVERSE_LABEL_MAP[int(v)] for v in predictions["naive_bayes"]],
            "naive_bayes_probability_spam": probabilities["naive_bayes"],
            "logistic_regression_prediction": [REVERSE_LABEL_MAP[int(v)] for v in predictions["logistic_regression"]],
            "logistic_regression_probability_spam": probabilities["logistic_regression"],
        }
    )
    predictions_df.to_csv(settings.predictions_path, index=False)

    metrics = {
        "project": "SMS Spam Detection Final Project",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "path": str(settings.data_path),
            "total_rows": int(len(df)),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "label_distribution": {k: int(v) for k, v in df["label"].value_counts().to_dict().items()},
            "test_size": test_size,
            "random_state": random_state,
            "max_features": max_features,
        },
        "models": ranked_models,
        "best_model": ranked_models[0]["model_key"],
        "mcnemar_test": mcnemar_stats,
        "artifacts": {
            "naive_bayes_model": str(settings.naive_bayes_model_path),
            "logistic_regression_model": str(settings.logistic_regression_model_path),
            "predictions": str(settings.predictions_path),
        },
    }

    with settings.metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    return metrics


def load_metrics() -> dict[str, Any]:
    if not settings.metrics_path.exists():
        raise ModelNotTrainedError("Model belum dilatih. Jalankan training dari dashboard terlebih dahulu.")
    with settings.metrics_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _load_models() -> dict[str, Pipeline]:
    if not settings.naive_bayes_model_path.exists() or not settings.logistic_regression_model_path.exists():
        raise ModelNotTrainedError("Model belum dilatih. Jalankan training dari dashboard terlebih dahulu.")
    return {
        "naive_bayes": joblib.load(settings.naive_bayes_model_path),
        "logistic_regression": joblib.load(settings.logistic_regression_model_path),
    }


def predict_message(message: str, threshold: float = 0.5) -> dict[str, Any]:
    models = _load_models()
    try:
        metrics = load_metrics()
        best_model_key = metrics.get("best_model")
    except Exception:
        best_model_key = None

    translated_message = message
    try:
        from deep_translator import GoogleTranslator
        # Menerjemahkan pesan otomatis ke Bahasa Inggris jika bukan bahasa Inggris
        translated = GoogleTranslator(source='auto', target='en').translate(message)
        if translated:
            translated_message = translated
    except Exception as exc:
        # Fallback ke pesan asli jika gagal
        print(f"Penerjemahan gagal: {exc}")

    prepared = prepare_text_for_model(translated_message)
    cleaned = clean_text(translated_message)
    predictions: list[dict[str, Any]] = []

    for model_key, model in models.items():
        spam_probability = float(model.predict_proba([prepared])[0][1])
        prediction_value = int(spam_probability >= threshold)
        predictions.append(
            {
                "model_name": "Multinomial Naive Bayes" if model_key == "naive_bayes" else "Logistic Regression",
                "model_key": model_key,
                "spam_probability": round(spam_probability, 6),
                "ham_probability": round(1.0 - spam_probability, 6),
                "prediction_label": REVERSE_LABEL_MAP[prediction_value],
                "prediction_value": prediction_value,
            }
        )

    if best_model_key:
        final_probability = next(item["spam_probability"] for item in predictions if item["model_key"] == best_model_key)
    else:
        final_probability = float(np.mean([item["spam_probability"] for item in predictions]))

    final_prediction_value = int(final_probability >= threshold)
    return {
        "raw_message": message,
        "translated_message": translated_message,
        "cleaned_message": cleaned,
        "threshold": threshold,
        "best_model": best_model_key,
        "final_prediction_label": REVERSE_LABEL_MAP[final_prediction_value],
        "final_spam_probability": round(final_probability, 6),
        "risk_level": risk_level(final_probability),
        "predictions": predictions,
    }
