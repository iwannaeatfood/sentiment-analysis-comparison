# build dataset and save train,val,test json files
# goemotions for train, val
# imdb for test

import json
from pathlib import Path

from datasets import load_dataset
from sklearn.model_selection import train_test_split

from data.label_mapping import convert_label
from data.clean import clean_text



# convert multi-label to binary label and clean text
def process_goemotions(dataset):
    features = dataset["train"].features["labels"]
    id2label = {i: name for i, name in enumerate(features.feature.names)}

    processed = []

    for split in ["train"]:
        for item in dataset[split]:
            label = convert_label(item["labels"], id2label)
            if label is None:
                continue

            text = clean_text(item["text"])
            if not text:
                continue

            processed.append({
                "text": text,
                "label": label
            })

    return processed


# imdb for  test
def process_imdb(dataset):
    processed = []

    for item in dataset["test"]:
        text = clean_text(item["text"])
        if not text:
            continue

        processed.append({
            "text": text,
            "label": int(item["label"])
        })

    return processed


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# count number of positive and negative samples
def count_labels(data):
    pos = sum(item["label"] == 1 for item in data)
    neg = sum(item["label"] == 0 for item in data)
    return pos, neg

# compute token length
def avg_length(data):
    return sum(len(item["text"].split()) for item in data) / len(data)


def length_stats(data):
    lengths = [len(item["text"].split()) for item in data]
    return min(lengths), max(lengths)


def print_report(train_data, val_data, test_data):
    train_pos, train_neg = count_labels(train_data)
    val_pos, val_neg = count_labels(val_data)
    test_pos, test_neg = count_labels(test_data)

    train_avg = avg_length(train_data)
    test_avg = avg_length(test_data)

    train_min, train_max = length_stats(train_data)
    test_min, test_max = length_stats(test_data)

    report = (
    f"summary of processed datasets (GoEmotions train,val, IMDb test)\n\n"
    f"train.json: {len(train_data)} samples (pos: {train_pos} / neg: {train_neg})\n"
    f"val.json: {len(val_data)} samples (pos: {val_pos} / neg: {val_neg})\n"
    f"test.json: {len(test_data)} samples (pos: {test_pos} / neg: {test_neg})\n"
    f"avg text length: train={train_avg:.1f} tokens, test={test_avg:.1f} tokens\n"

    f"shortest and longest texts in each split:\n"
    f"train length range: {train_min} - {train_max} tokens\n"
    f"test length range: {test_min} - {test_max} tokens\n"
    )
   
    

    print(report)
    return report


def save_report(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)



def main():
    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.json"
    val_path = output_dir / "val.json"
    test_path = output_dir / "test.json"

    if train_path.exists() and val_path.exists() and test_path.exists():
        train_data = load_json(train_path)
        val_data = load_json(val_path)
        test_data = load_json(test_path)
    else:
        cache_dir = Path("data/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)

        goemotions = load_dataset(
            "google-research-datasets/go_emotions",
            cache_dir=str(cache_dir)
        )

        imdb = load_dataset(
            "stanfordnlp/imdb",
            cache_dir=str(cache_dir)
        )

        goemotions_processed = process_goemotions(goemotions)
        imdb_processed = process_imdb(imdb)

        train_data, val_data = train_test_split(
            goemotions_processed,
            test_size=0.2,
            random_state=42,
            stratify=[item["label"] for item in goemotions_processed]
        )

        test_data = imdb_processed

        save_json(train_path, train_data)
        save_json(val_path, val_data)
        save_json(test_path, test_data)

    report = print_report(train_data, val_data, test_data)
    save_report(output_dir / "data_splits_report.txt", report)


if __name__ == "__main__":
    main()