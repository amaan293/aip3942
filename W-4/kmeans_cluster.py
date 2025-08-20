import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def kmeans(X, k, max_iters=100):
    np.random.seed()
    random_idx = np.random.choice(len(X), k, replace=False)
    centroids = X[random_idx]

    for _ in range(max_iters):
        clusters = [[] for _ in range(k)]
        labels = []
        for point in X:
            distances = [euclidean_distance(point, c) for c in centroids]
            cluster_idx = np.argmin(distances)
            clusters[cluster_idx].append(point)
            labels.append(cluster_idx)

        prev_centroids = centroids.copy()

        for idx in range(k):
            if clusters[idx]:
                centroids[idx] = np.mean(clusters[idx], axis=0)

        if np.all(prev_centroids == centroids):
            break

    return np.array(centroids), np.array(labels)

def compute_wcss(X, centroids, labels):
    wcss = 0
    for idx, point in enumerate(X):
        centroid = centroids[labels[idx]]
        
        wcss += np.sum((point - centroid) ** 2)
    return wcss

def silhouette_score(X, labels):
    n = len(X)
    unique_clusters = np.unique(labels)
    sil_scores = []

    for i in range(n):
        same_cluster = X[labels == labels[i]]
        other_clusters = [X[labels == c] for c in unique_clusters if c != labels[i]]

        if len(same_cluster) > 1:
            a_i = np.mean([euclidean_distance(X[i], p) for p in same_cluster if not np.array_equal(p, X[i])])
        else:
            a_i = 0

        b_i = np.min([np.mean([euclidean_distance(X[i], p) for p in cluster]) for cluster in other_clusters])
        s_i = (b_i - a_i) / max(a_i, b_i)
        sil_scores.append(s_i)

    return np.mean(sil_scores)

def find_optimal_k(X, K_range=range(2, 11), runs=5):
    mean_wcss = []
    std_wcss = []
    silhouette_scores = []

    for k in K_range:
        wcss_values = []
        sil_values = []

        for _ in range(runs):
            centroids, labels = kmeans(X, k)
            wcss = compute_wcss(X, centroids, labels)
            wcss_values.append(wcss)
            sil_values.append(silhouette_score(X, labels))

        mean_wcss.append(np.mean(wcss_values))
        std_wcss.append(np.std(wcss_values))
        silhouette_scores.append(np.mean(sil_values))

    return mean_wcss, std_wcss, silhouette_scores


def plot_elbow(K_range, mean_wcss, std_wcss):
    plt.errorbar(K_range, mean_wcss, yerr=std_wcss, fmt='-o', capsize=5)
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("WCSS (mean ± std)")
    plt.title("Elbow Method with WCSS Mean & Std")
    plt.show()


def plot_silhouette(K_range, silhouette_scores):
    plt.plot(K_range, silhouette_scores, marker='o')
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Mean Silhouette Score")
    plt.title("Silhouette Score vs k")
    plt.show()


def plot_final_clusters(X, final_centroids, final_labels, optimal_k):
    plt.scatter(X[:, 0], X[:, 1], c=final_labels, cmap='viridis', s=50, alpha=0.6)
    plt.scatter(final_centroids[:, 0], final_centroids[:, 1], c='red', marker='X', s=200, label="Centroids")
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.title(f"Final Clusters with k={optimal_k}")
    plt.legend()
    plt.show()


def main():
    data = pd.read_csv(r"D:\aip3942\W-4\Mall_Customers.csv")
    X = data.iloc[:, [3,4 ]].to_numpy()

    K_range = range(2, 11)
    mean_wcss, std_wcss, silhouette_scores = find_optimal_k(X, K_range)

    plot_elbow(K_range, mean_wcss, std_wcss)
    plot_silhouette(K_range, silhouette_scores)

    optimal_k = K_range[np.argmax(silhouette_scores)]

    print("\n Values ")
    for i, k in enumerate(K_range):
        print(f"k={k}: Mean WCSS={mean_wcss[i]:.2f}, Std WCSS={std_wcss[i]:.2f}, Silhouette={silhouette_scores[i]:.4f}")

    print(f"\nOptimal number of clusters (k): {optimal_k}")

    final_centroids, final_labels = kmeans(X, optimal_k)
    plot_final_clusters(X, final_centroids, final_labels, optimal_k)
main()
