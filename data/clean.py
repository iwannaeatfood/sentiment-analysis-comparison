# clean raw text by removing noise

import re


def clean_text(text):
    if not text:
        return ""

    # remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # remove Reddit  mentions
    text = re.sub(r"/(u|r)/\S+", "", text)

    # normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text