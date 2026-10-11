import torch
from transformers import pipeline

# Choose GPU if available, otherwise CPU.
device = 0 if torch.cuda.is_available() else -1

# Loading lightweight pre-trained models
classifier = pipeline(
    task="sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    tokenizer="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    truncation=True,
    max_length=512,
    device=device,
)

LABEL_MAP = {"NEGATIVE": 0, "POSITIVE": 1}

def predict_distilbert(data: list) -> list:
    predictions = []
    for idx, item in enumerate(data):
        result = classifier(item['text'])[0]
        pred = LABEL_MAP[result['label']]
        predictions.append({
            "id": idx,
            "text": item['text'],
            "true_label": item['label'],
            "predicted_label": pred,
            "confidence": round(result['score'], 4),
            "model": "distilbert-sst-2"
        })
    return predictions

# Configure the interface for Member E to call.
def predict_distilbert_single(text: str) -> dict:
    result = classifier(text)[0]
    pred = LABEL_MAP[result['label']]
    return {
        "label": pred,
        "confidence": round(result['score'], 4),
        "label_text": result['label'].lower()
    }