import argparse, json, sys
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.baseline import predict as predict_baseline_single
from models.pretrained_distilbert import predict_distilbert_single
from models.pretrained_roberta import predict_roberta_single

ID2LABEL = {0: "negative", 1: "positive"}
MAX_LENGTH = 128

# Define a predictor class for finetuned models for inference.
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

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Sentiment Analysis Prediction CLI Script")
    parser.add_argument("--text", type=str, help="Single text input for prediction")
    parser.add_argument("--file", type=str, help="Path to a text file with one sentence per line for batch prediction")
    parser.add_argument("--model", type=str, 
                        choices=["baseline", "distilbert", "roberta", "finetuned-distilbert", "finetuned-roberta"], 
                        default="finetuned-distilbert", help="Model to use for prediction")
    args = parser.parse_args()

    # Validate input
    if not args.text and not args.file:
        print("Error: Please provide either --text or --file for prediction.")
        parser.print_help()
        sys.exit(1)

    # Load the appropriate prediction function based on the selected model
    if args.model == "finetuned-distilbert":
        # Load the finetuned-distilbert model predictor. The path to the checkpoint may need to be adjusted if it's in a different location.
        predictor = FinetunedPredictor(str(Path(__file__).resolve().parent.parent / "models/distilbert-finetuned/best_model"))
        predict_func = predictor.predict
    elif args.model == "finetuned-roberta":
        predictor = FinetunedPredictor(str(Path(__file__).resolve().parent.parent / "models/roberta-finetuned/best_model"))
        predict_func = predictor.predict
    elif args.model == "baseline":
        # Ensure NLTK resources are available for the baseline model
        import nltk
        nltk.download("punkt")
        nltk.download("averaged_perceptron_tagger")
        nltk.download("wordnet")
        nltk.download("sentiwordnet")
        predict_func = predict_baseline_single
    elif args.model == "distilbert":
        predict_func = predict_distilbert_single
    elif args.model == "roberta":
        predict_func = predict_roberta_single

    # Read input texts
    texts = []
    if args.text:
        texts.append(args.text)
    elif args.file:
        with open(args.file, 'r') as f:
            texts = [line.strip() for line in f if line.strip()]
    
    # Run predictions
    for text in texts:
        result = predict_func(text)
        print("_" * 50)
        print(f"Text: {text:.100}{'' if len(text) <= 100 else '...'}")  # Print first 100 chars for brevity
        print(f"Predicted Label: {result['label']}")
        print(f"Label Text: {result['label_text']}")
        if "confidence" in result:
            print(f"Confidence: {result['confidence']}")
        print()

# Entry point
if __name__ == "__main__":
    main()