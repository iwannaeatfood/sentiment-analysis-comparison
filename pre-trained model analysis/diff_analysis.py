import json
import matplotlib.pyplot as plt

# 1. Configuration File Paths 
baseline_path = "predictions/baseline_predictions.json"
pretrained_path = "predictions/pretrained_predictions.json" 

def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
# 2. Precise Data Loading

try:
    # 1. Load Baseline 
    base_raw = load_data(baseline_path)
    baseline = list(base_raw.values())[0] if isinstance(base_raw, dict) else base_raw
    
    # 2. Load the Pretrained file containing both deep learning models
    pretrained_data = load_data(pretrained_path)
    
    # Print to confirm what models are actually inside the file
    print(f" Models contained in the Pretrained file: {list(pretrained_data.keys())}")
    
    # Precisely extract each model's data using its key
    roberta = pretrained_data.get("roberta", [])
    distilbert = pretrained_data.get("distilbert", [])
    
    if not distilbert:
        print(" Warning: 'distilbert' data not found in the file, please check the keys!")
        exit()

    print(f" Successfully loaded Baseline samples: {len(baseline)}")
    print(f" Successfully loaded RoBERTa samples: {len(roberta)}")
    print(f" Successfully loaded DistilBERT samples: {len(distilbert)}")
    
except Exception as e:
    print(f"Error loading files: {e}")
    exit()

# 3. Compare Predictions of the Three Models

diff_samples = []
total_samples = min(len(baseline), len(roberta), len(distilbert))

all_agree_count = 0
roberta_distilbert_agree_count = 0 

for i in range(total_samples):
    b, r, d = baseline[i], roberta[i], distilbert[i]

    b_pred = b.get("predicted_label")
    r_pred = r.get("predicted_label")
    d_pred = d.get("predicted_label")
    
    if b_pred == r_pred == d_pred:
        all_agree_count += 1
    else:
        diff_samples.append({
            "ID": b.get("id", i),
            "Text Snippet": str(b.get("text", ""))[:80] + "...",
            "True Label": b.get("true_label", "N/A"),
            "Baseline": b_pred,
            "RoBERTa": r_pred,
            "DistilBERT": d_pred
        })
        
        # Deep learning models reach a consensus, but differ from Baseline
        if r_pred == d_pred and r_pred != b_pred:
            roberta_distilbert_agree_count += 1

# 4. Print Statistics 
inconsistent_count = len(diff_samples)
other_discrepancy = inconsistent_count - roberta_distilbert_agree_count

print("\n" + "=" * 70)
print("STATISTICS SUMMARY (True 3-Way Comparison)")
print("=" * 70)
print(f"Total Samples: {total_samples}")
print(f"Complete Consensus (All 3 agree): {all_agree_count}")
print(f"Discrepancy (At least 1 differs): {inconsistent_count}")
print(f"Deep models agree, Baseline differs: {roberta_distilbert_agree_count}")
print(f"Deep models disagree (Mixed Disagreement): {other_discrepancy}")
print("=" * 70)

# Visualization 
plt.figure(figsize=(8, 8))
labels = [
    "All 3 Agree\n(Consensus)", 
    "RoBERTa & DistilBERT Agree\n(Baseline Differs)", 
    "Mixed Disagreement\n(Deep Models Disagree)"
]
sizes = [all_agree_count, roberta_distilbert_agree_count, other_discrepancy]
colors = ["#2ECC71", "#3498DB", "#E74C3C"]
explode = (0.05, 0.05, 0.1)  

# Filter out slices with size 0 to prevent charting errors/overlap
filtered_sizes, filtered_labels, filtered_colors, filtered_explode = [], [], [], []
for s, l, c, e in zip(sizes, labels, colors, explode):
    if s > 0:
        filtered_sizes.append(s)
        filtered_labels.append(l)
        filtered_colors.append(c)
        filtered_explode.append(e)

plt.pie(
    filtered_sizes, 
    explode=filtered_explode, 
    labels=filtered_labels, 
    autopct="%.2f%%", 
    colors=filtered_colors, 
    startangle=140
)

plt.title("Prediction Consistency: Baseline vs RoBERTa vs DistilBERT", pad=20, fontweight='bold')
plt.axis('equal') 
plt.tight_layout()
plt.savefig("true_result_3way.png", dpi=300)
plt.close()

print("\n  True 3-way model comparison visualization saved as true_result_3way.png")
