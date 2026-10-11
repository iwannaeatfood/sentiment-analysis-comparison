import json
from nltk.corpus import sentiwordnet as swn
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.tag import pos_tag

# Map NLTK POS tags to WordNet format for better accuracy
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def get_sentiment_score(text: str) -> int:
    # Tokenize text and perform POS tagging
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    
    pos_score, neg_score = 0.0, 0.0
    for token, tag in tagged:
        wn_tag = get_wordnet_pos(tag)
        if wn_tag is None:
            continue
        # Get sentiment scores from SentiWordNet
        synsets = list(swn.senti_synsets(token, pos=wn_tag))
        if synsets:
            # Use the first synset score (baseline simplification)
            best = synsets[0]
            pos_score += best.pos_score()
            neg_score += best.neg_score()
    
    # Return 1 for positive, 0 for negative
    return 1 if pos_score >= neg_score else 0

def predict_baseline(data: list) -> list:
    predictions = []
    for idx, item in enumerate(data):
        pred = get_sentiment_score(item['text'])
        predictions.append({
            "id": idx,
            "text": item['text'],
            "true_label": item['label'],
            "predicted_label": pred
        })
    return predictions

# Unified inference interface for Member E to use
def predict(text: str) -> dict:
    label = get_sentiment_score(text)
    return {
        "label": label,
        "confidence": 0.5,
        "label_text": "positive" if label == 1 else "negative"
    }