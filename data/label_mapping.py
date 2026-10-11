# mapping GoEmotions multi-labels to binary sentiment

# positive  categories
POSITIVE_LABELS = {
    'admiration', 'amusement', 'approval', 'caring',
    'desire', 'excitement', 'gratitude', 'joy',
    'love', 'optimism', 'pride', 'relief'
}

# negative categories
NEGATIVE_LABELS = {
    'anger', 'annoyance', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'fear', 'grief',
    'nervousness', 'remorse', 'sadness'
}


# discard neutral or ambiguous categories
NEUTRAL_LABELS = {'neutral', 'confusion', 'curiosity', 'surprise', 'realization'}


def convert_label(emotion_ids, id2label):
    if not emotion_ids:
        return None

    # map emotion ids to label names
    labels = {id2label[i] for i in emotion_ids if i in id2label}

    pos = labels & POSITIVE_LABELS
    neg = labels & NEGATIVE_LABELS

# return 1 for positive, 0 for negative

    if pos and not neg:
        return 1
    elif neg and not pos:
        return 0

    return None