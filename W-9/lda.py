import numpy as np

def lda_gibbs(corpus_ids, K, alpha=0.1, beta=0.01, n_iter=100, random_state=0, verbose=True):

    np.random.seed(random_state)
    D = len(corpus_ids)
    V = max(max(doc) for doc in corpus_ids) + 1

    doc_topic = np.zeros((D, K), dtype=int)
    topic_word = np.zeros((K, V), dtype=int)
    topic_count = np.zeros(K, dtype=int)

    z_assign = []
    for d, doc in enumerate(corpus_ids):
        zlist = []
        for w in doc:
            t = np.random.randint(K)
            zlist.append(t)
            doc_topic[d, t] += 1
            topic_word[t, w] += 1
            topic_count[t] += 1
        z_assign.append(zlist)

    for it in range(n_iter):
        for d, doc in enumerate(corpus_ids):
            for i, w in enumerate(doc):
                t = z_assign[d][i]
                doc_topic[d, t] -= 1
                topic_word[t, w] -= 1
                topic_count[t] -= 1

                left = doc_topic[d, :] + alpha
                right = (topic_word[:, w] + beta) / (topic_count + V * beta)
                p = left * right
                p /= p.sum()

                new_t = np.random.choice(K, p=p)
                z_assign[d][i] = new_t
                doc_topic[d, new_t] += 1
                topic_word[new_t, w] += 1
                topic_count[new_t] += 1

        if verbose and (it + 1) % 10 == 0:
            print(f"Iteration {it+1}/{n_iter}")

    theta = (doc_topic + alpha) / (doc_topic.sum(axis=1, keepdims=True) + K * alpha)
    phi = (topic_word + beta) / (topic_count[:, None] + V * beta)
    return theta, phi, z_assign

def perplexity(corpus_ids, theta, phi, eps=1e-12):

    N = sum(len(doc) for doc in corpus_ids)
    ll = 0.0
    for d, doc in enumerate(corpus_ids):
        th = theta[d]
        for w in doc:
            ll += np.log(np.dot(th, phi[:, w]) + eps)
    return np.exp(-ll / N)

def show_top_words(phi, vocab_rev, topn=10):
    K = phi.shape[0]
    for k in range(K):
        top_idx = np.argsort(-phi[k])[:topn]
        words = [vocab_rev[i] for i in top_idx]
        print(f"Topic {k}: {words}")
