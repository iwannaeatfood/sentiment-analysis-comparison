import gradio as gr
import argparse, json, sys
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.baseline import predict as predict_baseline_single
from models.pretrained_distilbert import predict_distilbert_single
from models.pretrained_roberta import predict_roberta_single


ID2LABEL = {0: "negative", 1: "positive"}
MAX_LENGTH = 128

class FinetunedPredictor:
    def __init__(self, model_path="models/distilbert-finetuned/best_model"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()

    def predict(self, text: str) -> dict:
        import torch
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]
        pred  = torch.argmax(probs).item()
        return {
            "label": pred,
            "confidence": round(probs[pred].item(), 4),
            "label_text": ID2LABEL[pred],
        }
    
distilbert_finetuned_predictor = FinetunedPredictor(str(Path(__file__).resolve().parent.parent / "models/distilbert-finetuned/best_model"))
roberta_finetuned_predictor = FinetunedPredictor(str(Path(__file__).resolve().parent.parent / "models/roberta-finetuned/best_model"))


def analyze(text, model):
    """
    Analyze the sentiment of the input text using the selected model.
    - model: "Baseline", "DistilBERT", "RoBERTa", "Finetuned DistilBERT", or "All Models"
        Returns a dictionary of results for the selected model(s).
    """
    results = {}
    if model == "All Models":
        results = {
            "Baseline": predict_baseline_single(text),
            "DistilBERT": predict_distilbert_single(text),
            "RoBERTa": predict_roberta_single(text),
            "Finetuned DistilBERT": distilbert_finetuned_predictor.predict(text),
            "Finetuned RoBERTa": roberta_finetuned_predictor.predict(text)
        }
    elif model == "Baseline":
        results = {"Baseline": predict_baseline_single(text)}
    elif model == "DistilBERT":
        results = {"DistilBERT": predict_distilbert_single(text)}
    elif model == "RoBERTa":
        results = {"RoBERTa": predict_roberta_single(text)}
    elif model == "Finetuned DistilBERT":
        results = {"Finetuned DistilBERT": distilbert_finetuned_predictor.predict(text)}
    elif model == "Finetuned RoBERTa":
        results = {"Finetuned RoBERTa": roberta_finetuned_predictor.predict(text)}
    return results


VOTE_MODELS = ["DistilBERT", "RoBERTa", "Finetuned DistilBERT", "Finetuned RoBERTa"]


def majority_vote(results: dict) -> dict:
    """
    Perform majority voting among all models in VOTE_MODELS.
    Returns a dictionary with the final label, label text, and vote counts.
    """
    votes = [results[m]["label"] for m in VOTE_MODELS if m in results]
    if not votes:
        return {}
    winner = 1 if votes.count(1) > votes.count(0) else 0
    pos_votes = votes.count(1)
    neg_votes = votes.count(0)
    return {
        "label": winner,
        "label_text": "positive" if winner == 1 else "negative",
        "pos_votes": pos_votes,
        "neg_votes": neg_votes,
    }


def format_results(results: dict):
    """
    Format the results into a markdown string for display in the Gradio interface.
    """
    if not results:
        return ""
    # If only one model's results, show a simple card.
    if len(results) == 1:
        model_name, r = next(iter(results.items()))
        conf = f"{r['confidence']:.1%}" if r["confidence"] != 0.5 else "N/A"
        return (
            f"### {model_name}\n\n"
            f"**{r['label_text'].upper()}**\n\n"
            f"Confidence: `{conf}`"
        )
    # If multiple models, show a comparison table + voting result.
    else:
        def set_row(name, r):
            conf = "N/A" if r["confidence"] == 0.5 else f"{r['confidence']:.1%}"
            return f"| {name} | {r['label_text'].capitalize()} | {conf} |"

        rows = "\n".join(set_row(name, r) for name, r in results.items())
        table = (
            "### All Models Comparison\n\n"
            "| Model | Prediction | Confidence |\n"
            "|-------|:----------:|:----------:|\n"
            + rows
        )

        # Perform majority voting among the 3 models (if available)
        vote = majority_vote(results)
        if not vote:
            return table

        # Add voting results section
        vote_section = (
            "\n\n---\n\n"
            "### Voting Result *(DistilBERT + RoBERTa + Finetuned DistilBERT + Finetuned RoBERTa)*\n\n"
            f"| Positive | Negative |\n"
            f"|:--------:|:--------:|\n"
            f"| {vote['pos_votes']} vote(s) | {vote['neg_votes']} vote(s) |\n\n"
            f"**Final Prediction: {vote['label_text'].upper()}**"
        )

        return table + vote_section


def predict(text: str, model: str):
    # Validate input
    if not text or not text.strip():
        return "Please enter some text to analyze."
    return format_results(analyze(text, model))


MODEL_CHOICES = ["Baseline", "DistilBERT", "RoBERTa", "Finetuned DistilBERT", "Finetuned RoBERTa", "All Models"]

# Example inputs for quick testing in the Gradio interface
EXAMPLES = [
    ["This movie was absolutely fantastic! I loved every moment.", "All Models"],
    ["The service was terrible and I will never come back.", "All Models"],
    ["The product is okay, nothing special.", "RoBERTa"],
]

with gr.Blocks(title="Sentiment Analysis Demo") as demo:
    gr.Markdown("# Sentiment Analysis Demo\nAnalyze text sentiment using one or all models.")

    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="Input Text",
                placeholder="Enter text to analyze...",
                lines=5,
            )
            model_dropdown = gr.Dropdown(
                choices=MODEL_CHOICES,
                value="All Models",
                label="Model",
            )
            submit_btn = gr.Button("Analyze", variant="primary")

        with gr.Column(scale=1):
            output = gr.Markdown(label="Results")

    gr.Examples(examples=EXAMPLES, inputs=[text_input, model_dropdown])

    submit_btn.click(fn=predict, inputs=[text_input, model_dropdown], outputs=output)
    text_input.submit(fn=predict, inputs=[text_input, model_dropdown], outputs=output)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True)

