import re

# Custom stopwords (minimal set, can expand)
CUSTOM_STOPWORDS = {
    "the", "is", "and", "in", "on", "at", "a", "an", "of", "for", "to", "with",
    "by", "from", "that", "this", "it", "as", "are", "was", "be", "or", "which",
    "but", "not"
}

# Simple rule-based lemmatizer
def simple_lemmatize(word):
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    elif word.endswith("ing") and len(word) > 4:
        return word[:-3]
    elif word.endswith("ed") and len(word) > 3:
        return word[:-2]
    elif word.endswith("s") and len(word) > 3:
        return word[:-1]
    else:
        return word

def preprocess_texts(raw_texts, min_token_length=3):

    processed_docs = []

    for doc in raw_texts:
        if not isinstance(doc, str):
            continue

        doc = doc.lower()                           # lowercase
        doc = re.sub(r"[^a-z\s]", " ", doc)        # remove non-letters
        words = doc.split()                         # tokenize

        toks = [
            simple_lemmatize(w)
            for w in words
            if w not in CUSTOM_STOPWORDS and len(w) >= min_token_length
        ]

        processed_docs.append(toks)

    return processed_docs
