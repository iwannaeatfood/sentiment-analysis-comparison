# Run script settings
import json
import os
DATA_PATH = "data/test.json"
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Test data does not exist, please check the path:{DATA_PATH}")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    test_data = json.load(f)
print(f" The test set was successfully loaded, containing {len(test_data)} samples.")

# Import the three saved model functions
from models.baseline import predict_baseline
from models.pretrained_roberta import predict_roberta
from models.pretrained_distilbert import predict_distilbert

# Run all models with one click
print("\nThe SentiWordNet baseline model is currently running.")
baseline_preds = predict_baseline(test_data)

print("Running RoBERTa zero-shot model")
roberta_preds = predict_roberta(test_data)

print("Running DistilBERT zero-shot model")
distilbert_preds = predict_distilbert(test_data)

# Save baseline results
BASELINE_OUTPUT = "predictions/baseline_predictions.json"
with open(BASELINE_OUTPUT, "w", encoding="utf-8") as f:
    json.dump(baseline_preds, f, indent=2, ensure_ascii=False)

# Save the results of the pre-trained model
PRETRAINED_OUTPUT = "predictions/pretrained_predictions.json"
with open(PRETRAINED_OUTPUT, "w", encoding="utf-8") as f:
    json.dump({
        "roberta": roberta_preds,
        "distilbert": distilbert_preds
    }, f, indent=2, ensure_ascii=False)

print(f"\n All predictions complete!")
print(f"Baseline results:{BASELINE_OUTPUT}")
print(f"Pre-training results:{PRETRAINED_OUTPUT}")