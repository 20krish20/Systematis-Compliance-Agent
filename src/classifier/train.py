"""
DistilBERT fine-tuning script for CFPB product/issue classification.
Logs all experiments to MLflow with accuracy, macro F1, and per-class metrics.
"""
from __future__ import annotations

import logging
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from src.classifier.distilbert_classifier import PRODUCT_LABELS
from src.config.settings import get_settings
from src.pipeline.data_ingestion import PRODUCT_TAXONOMY_MAP

logger = logging.getLogger(__name__)


class CFPBDataset(Dataset):
    def __init__(self, encodings: dict, labels: list[int]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": float((predictions == labels).mean()),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }


def train(
    data_path: str,
    output_dir: str = "models/checkpoints/distilbert_cfpb",
    num_epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    max_length: int = 256,
) -> None:
    cfg = get_settings()
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment_name)

    logger.info("Loading dataset from %s", data_path)
    df = pd.read_parquet(data_path)
    df = df[df["masked_text"].notna()]

    label_encoder = LabelEncoder()
    label_encoder.fit(PRODUCT_LABELS)

    # Normalize product labels
    def normalize(raw: str) -> str:
        for k, v in PRODUCT_TAXONOMY_MAP.items():
            if k.lower() in raw.lower():
                return v
        return "Other"

    df["product_normalized"] = df["metadata"].apply(
        lambda m: normalize(m.get("product_raw", "") if isinstance(m, dict) else "")
    )
    df = df[df["product_normalized"].isin(PRODUCT_LABELS)]
    df["label_id"] = label_encoder.transform(df["product_normalized"])

    texts = df["masked_text"].tolist()
    labels = df["label_id"].tolist()

    X_train, X_temp, y_train, y_temp = train_test_split(texts, labels, test_size=0.2, stratify=labels, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    train_enc = tokenizer(X_train, truncation=True, padding=True, max_length=max_length)
    val_enc = tokenizer(X_val, truncation=True, padding=True, max_length=max_length)
    test_enc = tokenizer(X_test, truncation=True, padding=True, max_length=max_length)

    train_dataset = CFPBDataset(train_enc, y_train)
    val_dataset = CFPBDataset(val_enc, y_val)
    test_dataset = CFPBDataset(test_enc, y_test)

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(PRODUCT_LABELS),
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_dir=f"{output_dir}/logs",
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    with mlflow.start_run(run_name="distilbert_cfpb_v1"):
        mlflow.log_params({
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_length": max_length,
            "train_size": len(X_train),
            "val_size": len(X_val),
            "test_size": len(X_test),
        })

        trainer.train()

        test_predictions = trainer.predict(test_dataset)
        test_preds = np.argmax(test_predictions.predictions, axis=-1)
        report = classification_report(y_test, test_preds, target_names=PRODUCT_LABELS, output_dict=True)

        mlflow.log_metric("test_accuracy", report["accuracy"])
        mlflow.log_metric("test_macro_f1", report["macro avg"]["f1-score"])
        for label in PRODUCT_LABELS:
            if label in report:
                mlflow.log_metric(f"f1_{label.replace('/', '_')}", report[label]["f1-score"])

        tokenizer.save_pretrained(output_dir)
        model.save_pretrained(output_dir)
        mlflow.log_artifacts(output_dir, artifact_path="model")

    logger.info("Training complete. Model saved to %s", output_dir)
