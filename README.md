# Sentiment Analysis Model Comparison (GoEmotions → IMDb)

UNSW COMP6713 (Natural Language Processing), Term 1 2026 — team project (5 members), April 2026.

**中文简介**：二分类情感分析（正面/负面），对比四类方法：SentiWordNet 词典基线、零样本 DistilBERT（SST-2）、零样本 RoBERTa、在 GoEmotions 上微调的 DistilBERT/RoBERTa，并在跨领域的 IMDb 影评测试集上评估，附 Gradio 演示。我负责 DistilBERT 微调流水线（Hugging Face Trainer + Weights & Biases 超参搜索）和团队共用的推理接口。

Original team repository (private, visible to team members only): `zkgao223/comp6713-sentiment-analysis`. This repository is a copy of the team code kept for my portfolio — model weights (`.safetensors`, `training_args.bin`) and the 25,000-review IMDb test file are not included. Teammates are referred to by role, not by name.

## My contribution

Source: the team's own `CONTRIBUTION.md` (private team repository) and the code in this repo.

| What | Where | Details |
|---|---|---|
| DistilBERT fine-tuning pipeline | `models/finetune_distilbert.py` | `distilbert-base-uncased` + Hugging Face `Trainer`; three run modes: `--debug` (CPU, 200 samples), default full training, `--sweep` |
| Hyperparameter search | same file, `SWEEP_CONFIG` | Weights & Biases **grid** search, 4 runs: learning rate {2e-5, 3e-5} × epochs {2, 3}; batch size 32 and weight decay 0.01 fixed; objective validation F1. Best: lr 3e-5, 2 epochs → **F1 0.929, accuracy 0.908** on the GoEmotions validation split |
| Shared inference interface | `FinetunedPredictor` in the same file | `predict(text)` → label (0/1), confidence, label text; loads checkpoint with `model.eval()` and `torch.no_grad()`; used by the CLI/Gradio demo and batch prediction written by teammates |
| Model distribution | Git LFS + cloud storage (see `.gitattributes`) | Weights shared with the team through Git LFS, Google Cloud Storage and a Drive backup; checked that `config.json` carries `id2label`/`label2id`/`num_labels` so downstream loaders work |
| Compute workaround | — | The GCP T4 GPU quota was unavailable, so training was moved to Google Colab |

The other four members, by function: data cleaning, label mapping and domain analysis; baseline and zero-shot model wrappers, batch inference and prediction-agreement analysis; RoBERTa fine-tuning, CLI and Gradio demo; evaluation, cross-domain testing, error analysis and report writing.

## Results (team, IMDb test set, 25,000 reviews)

From the team report. Precision/recall/F1 are for the positive class.

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| SentiWordNet baseline | 0.6107 | 0.5704 | 0.8964 | 0.6972 |
| RoBERTa (zero-shot) | 0.8223 | 0.9242 | 0.7022 | 0.7981 |
| DistilBERT-SST2 (zero-shot) | 0.8921 | 0.9151 | 0.8644 | 0.8890 |
| **DistilBERT (fine-tuned, my model)** | 0.7523 | 0.7197 | 0.8266 | 0.7695 |
| RoBERTa (fine-tuned) | 0.7744 | 0.7220 | 0.8925 | 0.7983 |

Two numbers that are easy to confuse: **F1 0.929 / accuracy 0.908 is on the in-domain GoEmotions validation split**, while **F1 0.77 / accuracy 0.75 is the same fine-tuned model on IMDb**. The best system overall was the zero-shot DistilBERT-SST2, not a fine-tuned model.

## Notes on the numbers

- **F1 0.929 / accuracy 0.908 is on the in-domain GoEmotions validation split** (65% positive: 3,318 positive / 1,803 negative); the same fine-tuned model scores F1 0.77 / accuracy 0.75 on IMDb. The best system overall was the zero-shot DistilBERT-SST2.
- **Domain shift:** training comments average about 13 tokens (max 31); IMDb reviews average about 225 tokens, and inputs are truncated at 128 tokens (`MAX_LENGTH`). The team report attributes the gap to domain mismatch; I did not run a separate experiment to isolate it.
- **Label mapping:** GoEmotions emotions are grouped into 12 positive and 11 negative classes; an example is kept only if it has positive or negative emotions but not both, so neutral-only and mixed-polarity comments are discarded (`data/label_mapping.py`).
- **Sweep scope:** a 2×2 grid (learning rate × epochs) run on Colab; batch size, warm-up and max length were not searched.

> Reproducibility status: the commands below come from the team's README. Pinned versions are in `requirements.txt` (Python 3.10/3.11). I have not re-run the whole pipeline on a clean machine for this copy; fine-tuning is meant for a GPU (the sweep was run on Google Colab) and needs `wandb login` for sweeps. `data/train.json` and `data/val.json` are included; `python -m data.build_dataset` regenerates all splits (including the 25,000-review IMDb test file) from Hugging Face datasets.

## Project Structure

```
comp6713-sentiment-analysis/
├── data/
│   ├── build_dataset.py          # Download and preprocess datasets
│   ├── clean.py                  # Text cleaning
│   └── label_mapping.py          # GoEmotions to binary label mapping
├── models/
│   ├── baseline.py               # SentiWordNet lexicon baseline
│   ├── pretrained_distilbert.py  # Zero-shot DistilBERT (SST-2)
│   ├── pretrained_roberta.py     # Zero-shot RoBERTa (Twitter)
│   ├── finetune_distilbert.py    # Fine-tune DistilBERT + W&B hyperparameter sweep
│   ├── finetune_roberta.ipynb    # Fine-tune RoBERTa + W&B hyperparameter sweep
│   ├── run_predictions.py        # Batch inference on test set
│   └── distilbert-finetuned/
│       └── best_model/           # Fine-tuned DistilBERT checkpoint (see below)
├── demo/
│   ├── app.py                    # Gradio web interface
│   └── predict.py                # CLI inference script
└── requirements.txt
```

## Setup

**Python 3.10 / 3.11**

```bash
pip install -r requirements.txt
```

For the baseline model, download NLTK data once:

```python
import nltk
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")
nltk.download("sentiwordnet")
```

## Fine-tuned Model Checkpoint Download

Required files in `models/distilbert-finetuned/best_model/`:
`config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`,
`special_tokens_map.json`, `vocab.txt`, `training_args.bin`

Required files in `models/roberta-finetuned/best_model/`:
`config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`,
`special_tokens_map.json`, `vocab.txt`, `training_args.bin`, `merges.txt`

### Recommended: Google Drive

Download both model checkpoints from the link below and place the files into the corresponding paths:

```
https://drive.google.com/drive/folders/13V4ibSTlBUi7hmaRvE6TnD9R1wCGuUWb?usp=sharing
```

### Alternative: Git LFS

The checkpoints are also tracked via Git LFS. After cloning, run:

```bash
git clone https://github.com/zkgao223/comp6713-sentiment-analysis.git
cd comp6713-sentiment-analysis
git lfs pull
```

Note: large files are not always downloaded automatically on clone, so `git lfs pull` may be required.

## Usage

**Prepare datasets** (downloads GoEmotions + IMDb automatically):

```bash
python -m data.build_dataset
```

**Domain analysis** (optional, for report analysis):

```bash
python -m data.domain_analysis
```

**Batch predictions on test set:**

```bash
python -m models.run_predictions
```

**Fine-tune DistilBERT** (requires `wandb login`; GPU recommended):

```bash
python -m models.finetune_distilbert --debug   # CPU, 200 samples
python -m models.finetune_distilbert           # full training
python -m models.finetune_distilbert --sweep   # W&B hyperparameter sweep (4 runs)
```

**Fine-tune RoBERTa** (requires `wandb login`; GPU recommended):

```bash
# Run the cells in models/finetune_roberta.ipynb
```

**CLI inference:**

```bash
python demo/predict.py --text "Great movie!" --model finetuned-distilbert
```

Three options available:

- `--text`: input string for prediction.
- `--file`: path to a file where each line is treated as a separate input. Either `--text` or `--file` is required; if both are provided, `--file` is ignored.
- `--model`: model to use. Accepted values: `baseline`, `distilbert`, `roberta`, `finetuned-roberta`, `finetuned-distilbert` (default).

**Gradio demo:**

```bash
python demo/app.py
```
