"""Predict sentiment for a single review with the trained RoBERTa model."""

from __future__ import annotations

from pathlib import Path
from typing import Dict


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model"
ID_TO_LABEL = {0: "Negative", 1: "Neutral", 2: "Positive"}


def predict_single_review(review_text: str, model_path: Path = MODEL_PATH) -> Dict[str, object]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    model.eval()

    inputs = tokenizer(review_text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1).squeeze(0)

    pred_id = int(torch.argmax(probabilities).item())
    return {
        "original_review": review_text,
        "predicted_sentiment": ID_TO_LABEL[pred_id],
        "confidence": round(float(probabilities[pred_id].item()), 4),
        "model_name": "roberta-base",
    }


if __name__ == "__main__":
    text = input("Review: ").strip()
    print(predict_single_review(text))
