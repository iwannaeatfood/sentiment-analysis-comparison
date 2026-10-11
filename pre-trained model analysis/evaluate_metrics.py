import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 1. Configuration File Paths
baseline_path = "predictions/baseline_predictions.json"
pretrained_path = "predictions/pretrained_predictions.json"

def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# 2. Precise Data Loading
print("Loading data and extracting labels...")
try:
    base_raw = load_data(baseline_path)
    baseline = list(base_raw.values())[0] if isinstance(base_raw, dict) else base_raw
    
    pretrained_data = load_data(pretrained_path)
    roberta = pretrained_data.get("roberta", [])
    distilbert = pretrained_data.get("distilbert", [])
    
except Exception as e:
    print(f"Error loading files: {e}")
    exit()

total_samples = min(len(baseline), len(roberta), len(distilbert))

# Arrays to hold the labels
y_true = []
y_base = []
y_rob = []
y_dist = []

for i in range(total_samples):
    b, r, d = baseline[i], roberta[i], distilbert[i]
    
    try:
        # We extract the true_label from the baseline dictionary
        t_label = int(b.get("true_label"))
        
        y_true.append(t_label)
        y_base.append(int(b.get("predicted_label")))
        y_rob.append(int(r.get("predicted_label")))
        y_dist.append(int(d.get("predicted_label")))
    except (ValueError, TypeError):
        continue  # Skip unparseable or missing data

print(f"Successfully extracted {len(y_true)} valid label pairs for calculation.\n")
# 3. Calculate Core Metrics
def get_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    return acc, p, r, f1

base_acc, base_p, base_r, base_f1 = get_metrics(y_true, y_base)
dist_acc, dist_p, dist_r, dist_f1 = get_metrics(y_true, y_dist)
rob_acc, rob_p, rob_r, rob_f1 = get_metrics(y_true, y_rob)

# 4. Print Professional Console Report
print("=" * 60)
print(" Final Model Performance Evaluation Report")
print("=" * 60)

print("\n[ NLTK Baseline Model ]")
print(f"  Accuracy:  {base_acc:.4f}  ({base_acc*100:.2f}%)")
print(f"  F1-Score:  {base_f1:.4f}")

print("\n[ DistilBERT Pre-trained Model ]")
print(f"  Accuracy:  {dist_acc:.4f}  ({dist_acc*100:.2f}%)")
print(f"  F1-Score:  {dist_f1:.4f}")

print("\n[ RoBERTa Pre-trained Model ]")
print(f"  Accuracy:  {rob_acc:.4f}  ({rob_acc*100:.2f}%)")
print(f"  F1-Score:  {rob_f1:.4f}")

print("\n" + "=" * 60)
print("  Performance Highlights:")
print(f"  DistilBERT vs Baseline (F1 Gain): +{(dist_f1 - base_f1):.4f}")
print(f"  RoBERTa vs Baseline (F1 Gain):    +{(rob_f1 - base_f1):.4f}")
print(f"  RoBERTa vs DistilBERT (F1 Gain):  +{(rob_f1 - dist_f1):.4f}")
print("=" * 60)

# 5. Plot and Save Comparison Chart
labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
base_metrics = [base_acc, base_p, base_r, base_f1]
dist_metrics = [dist_acc, dist_p, dist_r, dist_f1]
rob_metrics = [rob_acc, rob_p, rob_r, rob_f1]

x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots(figsize=(11, 7))

# Draw the 3 groups of bars
rects1 = ax.bar(x - width, base_metrics, width, label='NLTK Baseline', color='#7F8C8D')
rects2 = ax.bar(x, dist_metrics, width, label='DistilBERT', color='#F39C12')
rects3 = ax.bar(x + width, rob_metrics, width, label='RoBERTa', color='#2980B9')

# Add numerical labels on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

# Chart styling
ax.set_ylabel('Score', fontweight='bold')
ax.set_title('Performance Evaluation: Baseline vs DistilBERT vs RoBERTa', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylim(0, 1.15) # Leave space for the labels on top
ax.legend(loc='lower right', fontsize=10)
ax.grid(axis='y', linestyle='--', alpha=0.6)

# Save chart
fig.tight_layout()
plt.savefig("final_metrics_comparison.png", dpi=300)
print("\n  Final performance comparison chart saved as 'final_metrics_comparison.png'")