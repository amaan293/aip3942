import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups

from preprocess_lda import preprocess_texts
from map_lda import build_vocab, docs_to_id_lists
from lda import lda_gibbs, perplexity, show_top_words

def main():
    n_docs = 1000
    vocab_size = 2000
    ks = [5, 10, 15, 20]
    n_iter = 50

    print("Loading dataset...")
    data = fetch_20newsgroups(subset="train", remove=("headers","footers","quotes")).data[:n_docs]

    print("Preprocessing...")
    toks = preprocess_texts(data)

    print("Building vocab...")
    vocab = build_vocab(toks, vocab_size)
    vocab_rev = {i: w for w, i in vocab.items()}

    print("Converting docs...")
    corpus_ids = docs_to_id_lists(toks, vocab)
    print(f"Using {len(corpus_ids)} docs")

    perps = []
    for k in ks:
        print(f"\nRunning LDA with K={k}")
        theta, phi, _ = lda_gibbs(corpus_ids, K=k, n_iter=n_iter, alpha=0.1, beta=0.01)
        p = perplexity(corpus_ids, theta, phi)
        perps.append(p)
        print(f"Perplexity={p:.2f}")
        show_top_words(phi, vocab_rev, topn=8)

    plt.plot(ks, perps, marker="o")
    plt.xlabel("K")
    plt.ylabel("Perplexity")
    plt.title("Perplexity vs K")
    plt.show()

if __name__ == "__main__":
    main()
