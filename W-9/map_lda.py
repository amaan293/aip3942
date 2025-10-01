from collections import Counter

def build_vocab(docs_tokens, vocab_size=None):

    cnt = Counter()
    for doc in docs_tokens:
        cnt.update(doc)

    if vocab_size is None:
        items = list(cnt.keys())
    else:
        items = [w for w, _ in cnt.most_common(vocab_size)]

    vocab = {w: i for i, w in enumerate(items)}
    return vocab

def docs_to_id_lists(docs_tokens, vocab):

    out = []
    for doc in docs_tokens:
        ids = [vocab[w] for w in doc if w in vocab]
        if ids:
            out.append(ids)
    return out
