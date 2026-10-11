import torch
from transformers import pipeline

# Choose GPU if available, otherwise CPU.
device = 0 if torch.cuda.is_available() else -1

# Load the pre-trained RoBERTa sentiment classification model
classifier = pipeline(
    task="sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment",
    tokenizer="cardiffnlp/twitter-roberta-base-sentiment",
    truncation=True,
    max_length=512,
    device=device,
)

# Tag mapping
LABEL_MAP = {"LABEL_0": 0, "LABEL_1": 0, "LABEL_2": 1}
LABEL_TEXT_MAP = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}

def predict_roberta(data: list) -> list:
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
            "model": "roberta-base-sentiment"
        })
    return predictions

# Configure the interface for Member E to call.
def predict_roberta_single(text: str) -> dict:
    result = classifier(text)[0]
    pred = LABEL_MAP[result['label']]
    return {
        "label": pred,
        "confidence": round(result['score'], 4),
        "label_text": LABEL_TEXT_MAP[result['label']]
    }