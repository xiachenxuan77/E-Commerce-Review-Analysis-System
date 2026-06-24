"""Train a RoBERTa sentiment classifier on final_corpus_8w.csv."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "final_corpus.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"
CACHE_DIR = BASE_DIR / "hf_cache"

LABELS = ["Negative", "Neutral", "Positive"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}


@dataclass
class Config:
    model_name: str = "roberta-base"
    input_csv: Path = INPUT_CSV
    output_dir: Path = OUTPUT_DIR
    model_dir: Path = MODEL_DIR
    cache_dir: Path = CACHE_DIR
    test_size: float = 0.2
    valid_size: float = 0.1
    random_state: int = 42
    seed: int = 42
    max_length: int = 128
    train_batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 2e-5
    epochs: int = 3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06


def normalize_text(text: object) -> str:
    text = "" if pd.isna(text) else str(text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    return " ".join(text.split())


def load_corpus(path: Path) -> pd.DataFrame:
    required = ["review_id", "original_review", "cleaned_review", "true_sentiment"]
    df = pd.read_csv(path)
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError("Missing required columns: %s" % ", ".join(missing))

    df = df[required].copy()
    df["original_review"] = df["original_review"].fillna("").map(normalize_text)
    df["cleaned_review"] = df["cleaned_review"].fillna("")
    df["text"] = df["original_review"].where(
        df["original_review"].str.strip() != "",
        df["cleaned_review"],
    )
    df = df[df["text"].str.strip() != ""].copy()
    df["label"] = df["true_sentiment"].map(LABEL_TO_ID)

    if df["label"].isna().any():
        unknown = sorted(df.loc[df["label"].isna(), "true_sentiment"].dropna().unique())
        raise ValueError("Unsupported labels found: %s" % ", ".join(map(str, unknown)))

    df["label"] = df["label"].astype(int)
    return df.reset_index(drop=True)


def split_data(df: pd.DataFrame, config: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(
        df,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=df["label"],
    )
    train_df, valid_df = train_test_split(
        train_df,
        test_size=config.valid_size,
        random_state=config.random_state,
        stratify=train_df["label"],
    )
    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True), test_df.reset_index(drop=True)


def class_weights(labels: pd.Series) -> np.ndarray:
    counts = labels.value_counts().sort_index()
    return (counts.sum() / (len(counts) * counts)).to_numpy(dtype=np.float32)


def write_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(
        xticks=np.arange(len(LABELS)),
        yticks=np.arange(len(LABELS)),
        xticklabels=LABELS,
        yticklabels=LABELS,
        ylabel="True sentiment",
        xlabel="Predicted sentiment",
        title="RoBERTa Sentiment Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, format(matrix[row, col], "d"), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def train(config: Config = Config()) -> Dict[str, object]:
    import torch
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    from torch.utils.data import Dataset

    os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".matplotlib"))
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.model_dir.mkdir(parents=True, exist_ok=True)
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)

    df = load_corpus(config.input_csv)
    train_df, valid_df, test_df = split_data(df, config)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, cache_dir=str(config.cache_dir))

    class ReviewDataset(Dataset):
        def __init__(self, frame: pd.DataFrame):
            self.texts = frame["text"].tolist()
            self.labels = frame["label"].tolist()

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, index: int) -> Dict[str, object]:
            item = tokenizer(
                self.texts[index],
                truncation=True,
                max_length=config.max_length,
            )
            item["labels"] = self.labels[index]
            return item

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=len(LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        cache_dir=str(config.cache_dir),
    )

    weights = torch.tensor(class_weights(train_df["label"]), dtype=torch.float)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss_fn = torch.nn.CrossEntropyLoss(weight=weights.to(outputs.logits.device))
            loss = loss_fn(outputs.logits.view(-1, model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        report = classification_report(labels, preds, target_names=LABELS, output_dict=True, zero_division=0)
        return {
            "accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro"),
            "weighted_f1": f1_score(labels, preds, average="weighted"),
            "neutral_f1": report["Neutral"]["f1-score"],
            "neutral_precision": report["Neutral"]["precision"],
            "neutral_recall": report["Neutral"]["recall"],
        }

    training_args = TrainingArguments(
        output_dir=str(config.model_dir / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        num_train_epochs=config.epochs,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to=[],
        seed=config.seed,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=ReviewDataset(train_df),
        eval_dataset=ReviewDataset(valid_df),
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    test_dataset = ReviewDataset(test_df)
    eval_metrics = trainer.evaluate(test_dataset)
    test_output = trainer.predict(test_dataset)
    test_pred_ids = np.argmax(test_output.predictions, axis=-1)
    test_labels = test_df["label"].to_numpy()

    report = classification_report(test_labels, test_pred_ids, target_names=LABELS, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().reset_index().rename(columns={"index": "label"}).to_csv(
        config.output_dir / "classification_report.csv",
        index=False,
    )
    write_confusion_matrix(test_labels, test_pred_ids, config.output_dir / "confusion_matrix.png")

    all_output = trainer.predict(ReviewDataset(df))
    all_pred_ids = np.argmax(all_output.predictions, axis=-1)
    all_probabilities = torch.softmax(torch.tensor(all_output.predictions), dim=-1).numpy()

    results_df = df[["review_id", "original_review", "true_sentiment"]].copy()
    results_df["predicted_sentiment"] = [ID_TO_LABEL[int(idx)] for idx in all_pred_ids]
    results_df["confidence"] = np.max(all_probabilities, axis=-1)
    results_df["model_name"] = config.model_name
    results_df.to_csv(config.output_dir / "sentiment_results.csv", index=False)
    results_df[results_df["true_sentiment"] != results_df["predicted_sentiment"]].to_csv(
        config.output_dir / "error_analysis.csv",
        index=False,
    )

    model_path = config.model_dir / "best_model"
    comparison = {
        "model_name": config.model_name,
        "accuracy": eval_metrics.get("eval_accuracy"),
        "macro_f1": eval_metrics.get("eval_macro_f1"),
        "weighted_f1": eval_metrics.get("eval_weighted_f1"),
        "neutral_f1": eval_metrics.get("eval_neutral_f1"),
        "neutral_precision": eval_metrics.get("eval_neutral_precision"),
        "neutral_recall": eval_metrics.get("eval_neutral_recall"),
        "model_path": str(model_path),
    }
    pd.DataFrame([comparison]).to_csv(config.output_dir / "model_comparison.csv", index=False)
    pd.DataFrame(
        [
            {
                "model_name": config.model_name,
                "max_length": config.max_length,
                "learning_rate": config.learning_rate,
                "batch_size": config.train_batch_size,
                "epochs": config.epochs,
                "loss_type": "weighted_ce",
            }
        ]
    ).to_csv(config.output_dir / "best_params.csv", index=False)
    (config.output_dir / "training_summary.json").write_text(
        json.dumps({"model_name": config.model_name, "metrics": comparison}, indent=2),
        encoding="utf-8",
    )

    trainer.save_model(str(model_path))
    tokenizer.save_pretrained(str(model_path))
    return {"model_name": config.model_name, "model_path": str(model_path), "metrics": comparison}


def main() -> None:
    result = train()
    print("Best model: %s" % result["model_name"])
    print("Saved model: %s" % result["model_path"])
    print("Metrics: %s" % result["metrics"])
    print("Outputs saved to: %s" % OUTPUT_DIR)


if __name__ == "__main__":
    main()
