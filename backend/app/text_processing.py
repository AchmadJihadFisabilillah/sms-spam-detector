from __future__ import annotations

import re
from typing import Iterable

SPAM_KEYWORDS = {
    "free", "win", "winner", "won", "cash", "prize", "claim", "urgent",
    "offer", "limited", "call", "text", "reply", "selected", "reward",
    "bonus", "guaranteed", "congratulations", "click", "mobile", "txt",
}


def clean_text(text: str) -> str:
    """Normalize SMS text for consistent model input."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " urltoken ", text)
    text = re.sub(r"\b\d{5,}\b", " longnumbertoken ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _length_bucket(cleaned_text: str) -> str:
    word_count = len(cleaned_text.split())
    if word_count <= 6:
        return "length_short"
    if word_count <= 18:
        return "length_medium"
    return "length_long"


def _keyword_tokens(cleaned_text: str, keywords: Iterable[str] = SPAM_KEYWORDS) -> list[str]:
    words = set(cleaned_text.split())
    return [f"keyword_{keyword}" for keyword in keywords if keyword in words]


def prepare_text_for_model(text: str) -> str:
    """Clean text and append engineered tokens used by both ML pipelines."""
    raw = str(text)
    cleaned = clean_text(raw)
    tokens: list[str] = []

    if re.search(r"http\S+|www\S+", raw.lower()):
        tokens.append("contains_url")
    if re.search(r"\d", raw):
        tokens.append("contains_number")

    tokens.append(_length_bucket(cleaned))
    tokens.extend(_keyword_tokens(cleaned))

    return " ".join([cleaned, *tokens]).strip()


def risk_level(probability: float) -> str:
    if probability >= 0.85:
        return "High Spam Risk"
    if probability >= 0.65:
        return "Medium Spam Risk"
    if probability >= 0.45:
        return "Low-Medium Spam Risk"
    return "Low Spam Risk"
