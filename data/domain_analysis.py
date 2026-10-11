import json
import math
import re
from collections import Counter
from pathlib import Path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# lowercase the text and split it into word tokens
def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


# count numbers of positive and negative samples in one split
def get_label_counts(data):
    pos = sum(1 for item in data if item["label"] == 1)
    neg = sum(1 for item in data if item["label"] == 0)
    total = len(data)
    return {
        "total": total,
        "positive": pos,
        "negative": neg,
        "positive_ratio": pos / total if total else 0.0,
        "negative_ratio": neg / total if total else 0.0,
    }


def get_lengths(data):
    return [len(tokenize(item["text"])) for item in data]


# compute descriptive statistics of text length
def get_length_stats(data):
    lengths = get_lengths(data)
    if not lengths:
        return {
            "count": 0,
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "q1": 0.0,
            "q3": 0.0,
            "p90": 0.0,
        }

    lengths_sorted = sorted(lengths)
    n = len(lengths_sorted)
    mean = sum(lengths_sorted) / n

    if n % 2 == 1:
        median = lengths_sorted[n // 2]
    else:
        median = (lengths_sorted[n // 2 - 1] + lengths_sorted[n // 2]) / 2

    variance = sum((x - mean) ** 2 for x in lengths_sorted) / n
    std = math.sqrt(variance)

# compute percentile values
    def percentile(values, p):
        if not values:
            return 0.0
        k = (len(values) - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return float(values[int(k)])
        d0 = values[f] * (c - k)
        d1 = values[c] * (k - f)
        return float(d0 + d1)

    return {
        "count": n,
        "min": lengths_sorted[0],
        "max": lengths_sorted[-1],
        "mean": mean,
        "median": median,
        "std": std,
        "q1": percentile(lengths_sorted, 0.25),
        "q3": percentile(lengths_sorted, 0.75),
        "p90": percentile(lengths_sorted, 0.90),
    }


# build  word frequency counter
def get_vocab_counter(data):
    counter = Counter()
    for item in data:
        counter.update(tokenize(item["text"]))
    return counter


#  compute vocabulary level statistics
def get_vocab_stats(data):
    vocab = get_vocab_counter(data)
    total_tokens = sum(vocab.values())
    unique_tokens = len(vocab)
    # add lexical richness signals 
    hapax_count = sum(1 for _, count in vocab.items() if count == 1)
    avg_token_frequency = total_tokens / unique_tokens if unique_tokens else 0.0
    return {
        "total_tokens": total_tokens,
        "unique_tokens": unique_tokens,
        "type_token_ratio": unique_tokens / total_tokens if total_tokens else 0.0,
        "top_20": vocab.most_common(20),
        "hapax_count": hapax_count,
        "hapax_ratio": hapax_count / unique_tokens if unique_tokens else 0.0,
        "avg_token_frequency": avg_token_frequency,
    }


# compare train and test vocabularies
def get_overlap_stats(train_data, test_data):
    train_vocab = set(get_vocab_counter(train_data).keys())
    test_vocab = set(get_vocab_counter(test_data).keys())

    shared = train_vocab & test_vocab
    train_only = train_vocab - test_vocab
    test_only = test_vocab - train_vocab
    # compute Jaccard overlap measure symmetric vocabulary similarity
    union = train_vocab | test_vocab

    return {
        "train_vocab_size": len(train_vocab),
        "test_vocab_size": len(test_vocab),
        "shared_vocab_size": len(shared),
        "train_only_vocab_size": len(train_only),
        "test_only_vocab_size": len(test_only),
        "train_vocab_overlap_ratio": len(shared) / len(train_vocab) if train_vocab else 0.0,
        "test_vocab_overlap_ratio": len(shared) / len(test_vocab) if test_vocab else 0.0,
        "jaccard_overlap": len(shared) / len(union) if union else 0.0,
        "top_20_train_only": Counter({w: 1 for w in sorted(train_only)}).most_common(20),
        "top_20_test_only": Counter({w: 1 for w in sorted(test_only)}).most_common(20),
    }


# extract the most frequent words appear only in train or only in test
def get_top_unique_words(train_data, test_data, top_n=20):
    train_counter = get_vocab_counter(train_data)
    test_counter = get_vocab_counter(test_data)

    train_only = [(word, count) for word, count in train_counter.items() if word not in test_counter]
    test_only = [(word, count) for word, count in test_counter.items() if word not in train_counter]

    train_only = sorted(train_only, key=lambda x: (-x[1], x[0]))[:top_n]
    test_only = sorted(test_only, key=lambda x: (-x[1], x[0]))[:top_n]

    return {
        "top_train_only_words": train_only,
        "top_test_only_words": test_only,
    }

# measure out-of-vocabulary coverage of the target domain against the source domain
def get_oov_stats(source_data, target_data):
    source_vocab = set(get_vocab_counter(source_data).keys())
    target_counter = get_vocab_counter(target_data)

    if not target_counter:
        return {
            "target_total_tokens": 0,
            "target_unique_tokens": 0,
            "oov_unique_tokens": 0,
            "oov_total_occurrences": 0,
            "oov_unique_ratio": 0.0,
            "oov_token_ratio": 0.0,
            "top_20_oov_words": [],
        }

# identify words appear in target but not in source
    oov_items = [(word, count) for word, count in target_counter.items() if word not in source_vocab]
    oov_unique_tokens = len(oov_items)
    oov_total_occurrences = sum(count for _, count in oov_items)
    target_total_tokens = sum(target_counter.values())
    target_unique_tokens = len(target_counter)

    top_20_oov_words = sorted(oov_items, key=lambda x: (-x[1], x[0]))[:20]

    return {
        "target_total_tokens": target_total_tokens,
        "target_unique_tokens": target_unique_tokens,
        "oov_unique_tokens": oov_unique_tokens,
        "oov_total_occurrences": oov_total_occurrences,
        "oov_unique_ratio": oov_unique_tokens / target_unique_tokens if target_unique_tokens else 0.0,
        "oov_token_ratio": oov_total_occurrences / target_total_tokens if target_total_tokens else 0.0,
        "top_20_oov_words": top_20_oov_words,
    }


# capture surface writing style differences across dataset splits
def get_text_style_stats(data):
    total_samples = len(data)
    if total_samples == 0:
        return {
            "avg_char_length": 0.0,
            "avg_sentence_count": 0.0,
            "avg_exclamation_count": 0.0,
            "avg_question_count": 0.0,
            "avg_uppercase_ratio": 0.0,
            "avg_digit_ratio": 0.0,
            "avg_punctuation_ratio": 0.0,
        }

    char_lengths = []
    sentence_counts = []
    exclamation_counts = []
    question_counts = []
    uppercase_ratios = []
    digit_ratios = []
    punctuation_ratios = []

    for item in data:
        text = item["text"]
        char_len = len(text)
        char_lengths.append(char_len)

        sentence_count = len(re.findall(r"[.!?]+", text))
        sentence_counts.append(sentence_count if sentence_count > 0 else 1)

        exclamation_counts.append(text.count("!"))
        question_counts.append(text.count("?"))

        # compute simple character level style ratios
        if char_len > 0:
            uppercase_count = sum(1 for ch in text if ch.isupper())
            digit_count = sum(1 for ch in text if ch.isdigit())
            punctuation_count = sum(1 for ch in text if re.match(r"[^\w\s]", ch))

            uppercase_ratios.append(uppercase_count / char_len)
            digit_ratios.append(digit_count / char_len)
            punctuation_ratios.append(punctuation_count / char_len)
        else:
            uppercase_ratios.append(0.0)
            digit_ratios.append(0.0)
            punctuation_ratios.append(0.0)

    return {
        "avg_char_length": sum(char_lengths) / total_samples,
        "avg_sentence_count": sum(sentence_counts) / total_samples,
        "avg_exclamation_count": sum(exclamation_counts) / total_samples,
        "avg_question_count": sum(question_counts) / total_samples,
        "avg_uppercase_ratio": sum(uppercase_ratios) / total_samples,
        "avg_digit_ratio": sum(digit_ratios) / total_samples,
        "avg_punctuation_ratio": sum(punctuation_ratios) / total_samples,
    }


# bucket text lengths 
def get_length_bucket_stats(data):
    lengths = get_lengths(data)
    total = len(lengths)
    if total == 0:
        return {
            "0_20": 0,
            "21_50": 0,
            "51_100": 0,
            "101_200": 0,
            "201_plus": 0,
            "0_20_ratio": 0.0,
            "21_50_ratio": 0.0,
            "51_100_ratio": 0.0,
            "101_200_ratio": 0.0,
            "201_plus_ratio": 0.0,
        }

    buckets = {
        "0_20": 0,
        "21_50": 0,
        "51_100": 0,
        "101_200": 0,
        "201_plus": 0,
    }

    # group each sample into a length interval
    for length in lengths:
        if length <= 20:
            buckets["0_20"] += 1
        elif length <= 50:
            buckets["21_50"] += 1
        elif length <= 100:
            buckets["51_100"] += 1
        elif length <= 200:
            buckets["101_200"] += 1
        else:
            buckets["201_plus"] += 1

    results = dict(buckets)
    for key, value in buckets.items():
        results[f"{key}_ratio"] = value / total

    return results

# compare text length patterns
def get_label_conditioned_length_stats(data):
    positive_lengths = [len(tokenize(item["text"])) for item in data if item["label"] == 1]
    negative_lengths = [len(tokenize(item["text"])) for item in data if item["label"] == 0]

    def summarize(lengths):
        if not lengths:
            return {
                "count": 0,
                "mean": 0.0,
                "median": 0.0,
                "min": 0,
                "max": 0,
            }

        lengths_sorted = sorted(lengths)
        n = len(lengths_sorted)
        mean = sum(lengths_sorted) / n

        if n % 2 == 1:
            median = lengths_sorted[n // 2]
        else:
            median = (lengths_sorted[n // 2 - 1] + lengths_sorted[n // 2]) / 2

        return {
            "count": n,
            "mean": mean,
            "median": median,
            "min": lengths_sorted[0],
            "max": lengths_sorted[-1],
        }

    return {
        "positive": summarize(positive_lengths),
        "negative": summarize(negative_lengths),
    }

# aggregate domain shift indicators into one summary block
def get_domain_summary(train_length_stats, test_length_stats, overlap_stats, train_vocab_stats, test_vocab_stats, oov_stats, train_style_stats, test_style_stats):
    return {
        "test_vs_train_mean_length_ratio": (
            test_length_stats["mean"] / train_length_stats["mean"]
            if train_length_stats["mean"] else 0.0
        ),
        "test_vs_train_unique_token_ratio": (
            test_vocab_stats["unique_tokens"] / train_vocab_stats["unique_tokens"]
            if train_vocab_stats["unique_tokens"] else 0.0
        ),
        "vocab_jaccard_overlap": overlap_stats["jaccard_overlap"],
        "test_oov_token_ratio_against_train": oov_stats["oov_token_ratio"],
        "test_oov_unique_ratio_against_train": oov_stats["oov_unique_ratio"],
        "test_vs_train_sentence_count_ratio": (
            test_style_stats["avg_sentence_count"] / train_style_stats["avg_sentence_count"]
            if train_style_stats["avg_sentence_count"] else 0.0
        ),
        "test_vs_train_punctuation_ratio": (
            test_style_stats["avg_punctuation_ratio"] / train_style_stats["avg_punctuation_ratio"]
            if train_style_stats["avg_punctuation_ratio"] else 0.0
        ),
    }


def format_label_stats(name, stats):
    return (
        f"{name} label distribution\n"
        f"total: {stats['total']}\n"
        f"positive: {stats['positive']} ({stats['positive_ratio']:.4f})\n"
        f"negative: {stats['negative']} ({stats['negative_ratio']:.4f})\n"
    )


def format_length_stats(name, stats):
    return (
        f"{name} length statistics\n"
        f"count: {stats['count']}\n"
        f"min: {stats['min']}\n"
        f"max: {stats['max']}\n"
        f"mean: {stats['mean']:.2f}\n"
        f"median: {stats['median']:.2f}\n"
        f"std: {stats['std']:.2f}\n"
        f"q1: {stats['q1']:.2f}\n"
        f"q3: {stats['q3']:.2f}\n"
        f"p90: {stats['p90']:.2f}\n"
    )


def format_vocab_stats(name, stats):
    top_words = ", ".join([f"{word}:{count}" for word, count in stats["top_20"]])
    return (
        f"{name} vocabulary statistics\n"
        f"total_tokens: {stats['total_tokens']}\n"
        f"unique_tokens: {stats['unique_tokens']}\n"
        f"type_token_ratio: {stats['type_token_ratio']:.6f}\n"
        f"hapax_count: {stats['hapax_count']}\n"
        f"hapax_ratio: {stats['hapax_ratio']:.6f}\n"
        f"avg_token_frequency: {stats['avg_token_frequency']:.6f}\n"
        f"top_20_words: {top_words}\n"
    )


def format_overlap_stats(stats):
    train_only_words = ", ".join([f"{word}:{count}" for word, count in stats["top_train_only_words"]])
    test_only_words = ", ".join([f"{word}:{count}" for word, count in stats["top_test_only_words"]])

    return (
        f"train_vocab_size: {stats['train_vocab_size']}\n"
        f"test_vocab_size: {stats['test_vocab_size']}\n"
        f"shared_vocab_size: {stats['shared_vocab_size']}\n"
        f"train_only_vocab_size: {stats['train_only_vocab_size']}\n"
        f"test_only_vocab_size: {stats['test_only_vocab_size']}\n"
        f"train_vocab_overlap_ratio: {stats['train_vocab_overlap_ratio']:.6f}\n"
        f"test_vocab_overlap_ratio: {stats['test_vocab_overlap_ratio']:.6f}\n"
        f"jaccard_overlap: {stats['jaccard_overlap']:.6f}\n"
        f"top_train_only_words: {train_only_words}\n"
        f"top_test_only_words: {test_only_words}\n"
    )


def format_oov_stats(stats):
    top_oov_words = ", ".join([f"{word}:{count}" for word, count in stats["top_20_oov_words"]])
    return (
        f"test oov statistics against train\n"
        f"target_total_tokens: {stats['target_total_tokens']}\n"
        f"target_unique_tokens: {stats['target_unique_tokens']}\n"
        f"oov_unique_tokens: {stats['oov_unique_tokens']}\n"
        f"oov_total_occurrences: {stats['oov_total_occurrences']}\n"
        f"oov_unique_ratio: {stats['oov_unique_ratio']:.6f}\n"
        f"oov_token_ratio: {stats['oov_token_ratio']:.6f}\n"
        f"top_20_oov_words: {top_oov_words}\n"
    )


def format_text_style_stats(name, stats):
    return (
        f"{name} text style statistics\n"
        f"avg_char_length: {stats['avg_char_length']:.2f}\n"
        f"avg_sentence_count: {stats['avg_sentence_count']:.2f}\n"
        f"avg_exclamation_count: {stats['avg_exclamation_count']:.2f}\n"
        f"avg_question_count: {stats['avg_question_count']:.2f}\n"
        f"avg_uppercase_ratio: {stats['avg_uppercase_ratio']:.6f}\n"
        f"avg_digit_ratio: {stats['avg_digit_ratio']:.6f}\n"
        f"avg_punctuation_ratio: {stats['avg_punctuation_ratio']:.6f}\n"
    )


def format_length_bucket_stats(name, stats):
    return (
        f"{name} length bucket statistics\n"
        f"0_20: {stats['0_20']} ({stats['0_20_ratio']:.4f})\n"
        f"21_50: {stats['21_50']} ({stats['21_50_ratio']:.4f})\n"
        f"51_100: {stats['51_100']} ({stats['51_100_ratio']:.4f})\n"
        f"101_200: {stats['101_200']} ({stats['101_200_ratio']:.4f})\n"
        f"201_plus: {stats['201_plus']} ({stats['201_plus_ratio']:.4f})\n"
    )


def format_label_conditioned_length_stats(name, stats):
    return (
        f"{name} label conditioned length statistics\n"
        f"positive_count: {stats['positive']['count']}\n"
        f"positive_mean: {stats['positive']['mean']:.2f}\n"
        f"positive_median: {stats['positive']['median']:.2f}\n"
        f"positive_min: {stats['positive']['min']}\n"
        f"positive_max: {stats['positive']['max']}\n"
        f"negative_count: {stats['negative']['count']}\n"
        f"negative_mean: {stats['negative']['mean']:.2f}\n"
        f"negative_median: {stats['negative']['median']:.2f}\n"
        f"negative_min: {stats['negative']['min']}\n"
        f"negative_max: {stats['negative']['max']}\n"
    )


def format_domain_summary(stats):
    return (
        f"domain shift summary\n"
        f"test_vs_train_mean_length_ratio: {stats['test_vs_train_mean_length_ratio']:.6f}\n"
        f"test_vs_train_unique_token_ratio: {stats['test_vs_train_unique_token_ratio']:.6f}\n"
        f"vocab_jaccard_overlap: {stats['vocab_jaccard_overlap']:.6f}\n"
        f"test_oov_token_ratio_against_train: {stats['test_oov_token_ratio_against_train']:.6f}\n"
        f"test_oov_unique_ratio_against_train: {stats['test_oov_unique_ratio_against_train']:.6f}\n"
        f"test_vs_train_sentence_count_ratio: {stats['test_vs_train_sentence_count_ratio']:.6f}\n"
        f"test_vs_train_punctuation_ratio: {stats['test_vs_train_punctuation_ratio']:.6f}\n"
    )


def save_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    train_path = Path("data/train.json")
    val_path = Path("data/val.json")
    test_path = Path("data/test.json")

    train_data = load_json(train_path)
    val_data = load_json(val_path)
    test_data = load_json(test_path)


    # compute label distribution for each split
    train_label_stats = get_label_counts(train_data)
    val_label_stats = get_label_counts(val_data)
    test_label_stats = get_label_counts(test_data)

    # compute text length statistics
    train_length_stats = get_length_stats(train_data)
    val_length_stats = get_length_stats(val_data)
    test_length_stats = get_length_stats(test_data)

    # compute vocabulary statistics
    train_vocab_stats = get_vocab_stats(train_data)
    val_vocab_stats = get_vocab_stats(val_data)
    test_vocab_stats = get_vocab_stats(test_data)

    # compare train and test vocabularies to measure domain shift
    overlap_stats = get_overlap_stats(train_data, test_data)
    unique_word_stats = get_top_unique_words(train_data, test_data)

    overlap_stats["top_train_only_words"] = unique_word_stats["top_train_only_words"]
    overlap_stats["top_test_only_words"] = unique_word_stats["top_test_only_words"]

    train_style_stats = get_text_style_stats(train_data)
    val_style_stats = get_text_style_stats(val_data)
    test_style_stats = get_text_style_stats(test_data)

    train_length_bucket_stats = get_length_bucket_stats(train_data)
    val_length_bucket_stats = get_length_bucket_stats(val_data)
    test_length_bucket_stats = get_length_bucket_stats(test_data)

    train_label_conditioned_length_stats = get_label_conditioned_length_stats(train_data)
    val_label_conditioned_length_stats = get_label_conditioned_length_stats(val_data)
    test_label_conditioned_length_stats = get_label_conditioned_length_stats(test_data)

    test_oov_stats = get_oov_stats(train_data, test_data)

    domain_summary = get_domain_summary(
        train_length_stats,
        test_length_stats,
        overlap_stats,
        train_vocab_stats,
        test_vocab_stats,
        test_oov_stats,
        train_style_stats,
        test_style_stats,
    )

    report = "\n\n".join(
        [
            "domain analysis report",
            format_label_stats("train", train_label_stats),
            format_label_stats("val", val_label_stats),
            format_label_stats("test", test_label_stats),
            format_length_stats("train", train_length_stats),
            format_length_stats("val", val_length_stats),
            format_length_stats("test", test_length_stats),
            format_length_bucket_stats("train", train_length_bucket_stats),
            format_length_bucket_stats("val", val_length_bucket_stats),
            format_length_bucket_stats("test", test_length_bucket_stats),
            format_label_conditioned_length_stats("train", train_label_conditioned_length_stats),
            format_label_conditioned_length_stats("val", val_label_conditioned_length_stats),
            format_label_conditioned_length_stats("test", test_label_conditioned_length_stats),
            format_vocab_stats("train", train_vocab_stats),
            format_vocab_stats("val", val_vocab_stats),
            format_vocab_stats("test", test_vocab_stats),
            format_text_style_stats("train", train_style_stats),
            format_text_style_stats("val", val_style_stats),
            format_text_style_stats("test", test_style_stats),
            "vocabulary overlap statistics\n" + format_overlap_stats(overlap_stats),
            format_oov_stats(test_oov_stats),
            format_domain_summary(domain_summary),
        ]
    )

    output_dir = Path("data/analysis")
    save_text(output_dir / "domain_analysis_report.txt", report)

    results = {
        "train": {
            "label_stats": train_label_stats,
            "length_stats": train_length_stats,
            "length_bucket_stats": train_length_bucket_stats,
            "label_conditioned_length_stats": train_label_conditioned_length_stats,
            "vocab_stats": train_vocab_stats,
            "style_stats": train_style_stats,
        },
        "val": {
            "label_stats": val_label_stats,
            "length_stats": val_length_stats,
            "length_bucket_stats": val_length_bucket_stats,
            "label_conditioned_length_stats": val_label_conditioned_length_stats,
            "vocab_stats": val_vocab_stats,
            "style_stats": val_style_stats,
        },
        "test": {
            "label_stats": test_label_stats,
            "length_stats": test_length_stats,
            "length_bucket_stats": test_length_bucket_stats,
            "label_conditioned_length_stats": test_label_conditioned_length_stats,
            "vocab_stats": test_vocab_stats,
            "style_stats": test_style_stats,
            "oov_stats_against_train": test_oov_stats,
        },
        "train_test_overlap": overlap_stats,
        "domain_summary": domain_summary,
    }

    save_json(output_dir / "domain_analysis_results.json", results)

    print(report)


if __name__ == "__main__":
    main()