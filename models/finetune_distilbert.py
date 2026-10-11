"""
DistilBERT fine-tuning with W&B hyperparameter sweep.

Usage:
    python -m models.finetune_distilbert --debug   # CPU, 200 samples
    python -m models.finetune_distilbert           # full training
    python -m models.finetune_distilbert --sweep   # W&B grid search (4 runs)
"""

import argparse
import json
import os

import numpy as np
import wandb
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# Model and training constants
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
WANDB_PROJECT = "comp6713-sentiment"

ID2LABEL = {0: "negative", 1: "positive"}
LABEL2ID = {"negative": 0, "positive": 1}

# W&B sweep configuration: 2x2 grid search (4 runs total)
SWEEP_CONFIG = {
    "name": "distilbert-sweep",
    "method": "grid",
    "metric": {"name": "eval/f1", "goal": "maximize"},
    "parameters": {
        "learning_rate": {"values": [2e-5, 3e-5]},
        "num_train_epochs": {"values": [2, 3]},
        "per_device_train_batch_size": {"value": 32},
        "weight_decay": {"value": 0.01},
    },
}


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_datasets(train_path, val_path, debug=False):
    train_data = load_json(train_path)
    val_data = load_json(val_path)
    if debug:
        train_data = train_data[:200]
        val_data = val_data[:50]
        print(f"[DEBUG] Using {len(train_data)} train / {len(val_data)} val samples")
    else:
        print(f"[INFO] Train: {len(train_data)} | Val: {len(val_data)}")
    return Dataset.from_list(train_data), Dataset.from_list(val_data)


def tokenize_dataset(dataset, tokenizer):
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )
    return dataset.map(tokenize, batched=True, remove_columns=["text"])


def compute_metrics(eval_pred):
    preds = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="binary"),
    }


def train_with_config(train_ds, val_ds, config, save_dir, debug=False):
    """Train DistilBERT with the given hyperparameter config."""
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    run_name = (
        f"lr{config['learning_rate']:.0e}_ep{config['num_train_epochs']}"
        if not debug else "debug-run"
    )

    training_args = TrainingArguments(
        output_dir=f"./checkpoints/{run_name}",
        run_name=run_name,
        num_train_epochs=1 if debug else config["num_train_epochs"],
        per_device_train_batch_size=16 if debug else config["per_device_train_batch_size"],
        per_device_eval_batch_size=64,
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="wandb",
        logging_steps=10 if debug else 50,
        no_cuda=debug,
        fp16=not debug,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    return trainer


def train(debug=False):
    """Run a single training pass with fixed hyperparameters."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, "data", "train.json")
    val_path = os.path.join(base_dir, "data", "val.json")
    save_dir = os.path.join(base_dir, "models", "distilbert-finetuned", "best_model")

    wandb.init(
        project=WANDB_PROJECT,
        name="distilbert-debug" if debug else "distilbert-finetune-v1",
        config={
            "model": MODEL_NAME,
            "max_length": MAX_LENGTH,
            "learning_rate": 2e-5,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 32,
            "weight_decay": 0.01,
        },
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds, val_ds = build_datasets(train_path, val_path, debug=debug)
    train_ds = tokenize_dataset(train_ds, tokenizer)
    val_ds = tokenize_dataset(val_ds, tokenizer)

    config = {
        "learning_rate": 2e-5,
        "num_train_epochs": 3,
        "per_device_train_batch_size": 32,
        "weight_decay": 0.01,
    }

    print(f"\n{'[DEBUG] ' if debug else ''}Starting training...")
    trainer = train_with_config(train_ds, val_ds, config, save_dir, debug=debug)

    os.makedirs(save_dir, exist_ok=True)
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"\nModel saved to {save_dir}")
    wandb.finish()


def run_sweep():
    """
    Run a W&B grid search over learning_rate x num_train_epochs (4 runs).
    Results are logged to the W&B dashboard for comparison.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, "data", "train.json")
    val_path = os.path.join(base_dir, "data", "val.json")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds, val_ds = build_datasets(train_path, val_path, debug=False)
    train_ds = tokenize_dataset(train_ds, tokenizer)
    val_ds = tokenize_dataset(val_ds, tokenizer)

    def sweep_train():
        with wandb.init() as run:
            config = wandb.config
            save_dir = os.path.join(
                base_dir, "models", "distilbert-finetuned",
                f"sweep_{run.name}"
            )
            trainer = train_with_config(
                train_ds, val_ds,
                dict(config),
                save_dir,
                debug=False,
            )
            os.makedirs(save_dir, exist_ok=True)
            trainer.save_model(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"Run {run.name} saved to {save_dir}")

    sweep_id = wandb.sweep(SWEEP_CONFIG, project=WANDB_PROJECT)
    print(f"\nSweep ID: {sweep_id}")
    print(f"View at: https://wandb.ai/<your-username>/{WANDB_PROJECT}/sweeps/{sweep_id}\n")

    wandb.agent(sweep_id, function=sweep_train, count=4)

    print("\nSweep complete. Check W&B dashboard for best hyperparameters.")
    print(f"https://wandb.ai/<your-username>/{WANDB_PROJECT}/sweeps/{sweep_id}")


class FinetunedPredictor:
    """
    Inference interface for the fine-tuned DistilBERT model.

    Usage:
        predictor = FinetunedPredictor()
        result = predictor.predict("This movie was great!")
        # returns: {'label': 1, 'confidence': 0.97, 'label_text': 'positive'}
    """

    def __init__(self, model_path="models/distilbert-finetuned/best_model"):
        import torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.torch = torch

    def predict(self, text: str) -> dict:
        import torch
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]
        pred = torch.argmax(probs).item()
        return {
            "label": pred,
            "confidence": round(probs[pred].item(), 4),
            "label_text": ID2LABEL[pred],
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true",
                        help="run on CPU with 200 samples to verify the pipeline")
    parser.add_argument("--sweep", action="store_true",
                        help="run W&B hyperparameter sweep (4 runs, GPU recommended)")
    args = parser.parse_args()

    if args.sweep:
        run_sweep()
    else:
        train(debug=args.debug)
