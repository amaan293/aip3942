import numpy as np
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
from numba import njit
def accuracy_score(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

def f1_score(y_true, y_pred, average="macro"):
    classes = np.unique(y_true)
    f1s = []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1s.append(f1)
    return np.mean(f1s)


def gini_index(y):
    m = len(y)
    if m == 0:
        return 0
    probs = [np.sum(y == c) / m for c in np.unique(y)]
    return 1 - sum(p ** 2 for p in probs)


def entropy(y):
    m = len(y)
    if m == 0:
        return 0
    probs = [np.sum(y == c) / m for c in np.unique(y)]
    return -sum(p * np.log2(p) for p in probs if p > 0)

def best_split(X, y, criterion="gini", n_features=None):
    m, n = X.shape
    if m <= 1:
        return None, None
    if n_features is None:
        n_features = n
    features = np.random.choice(n, n_features, replace=False)
    best_gain, best_feat, best_thresh = -1, None, None
    current_impurity = gini_index(y) if criterion == "gini" else entropy(y)
    for feat in features:
        thresholds = np.unique(X[:, feat])
        for t in thresholds:
            left_idx = X[:, feat] <= t
            right_idx = ~left_idx
            if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
                continue
            left_impurity = gini_index(y[left_idx]) if criterion == "gini" else entropy(y[left_idx])
            right_impurity = gini_index(y[right_idx]) if criterion == "gini" else entropy(y[right_idx])
            gain = current_impurity - (len(y[left_idx]) / m) * left_impurity - (len(y[right_idx]) / m) * right_impurity
            if gain > best_gain:
                best_gain, best_feat, best_thresh = gain, feat, t
    return best_feat, best_thresh

def build_tree(X, y, max_depth=None, depth=0, criterion="gini", extra=False):
    if len(np.unique(y)) == 1:
        return np.unique(y)[0]
    if max_depth is not None and depth >= max_depth:
        return np.bincount(y).argmax()
    n_features = int(np.sqrt(X.shape[1])) if not extra else X.shape[1]
    feat, thresh = best_split(X, y, criterion, n_features)
    if feat is None:
        return np.bincount(y).argmax()
    left_idx = X[:, feat] <= thresh
    right_idx = ~left_idx
    left = build_tree(X[left_idx], y[left_idx], max_depth, depth + 1, criterion, extra)
    right = build_tree(X[right_idx], y[right_idx], max_depth, depth + 1, criterion, extra)
    return (feat, thresh, left, right)


def predict_tree(x, tree):
    if not isinstance(tree, tuple):
        return tree
    feat, thresh, left, right = tree
    return predict_tree(x, left if x[feat] <= thresh else right)

def fit_forest(X, y, n_trees=10, max_depth=None, criterion="gini", extra=False):
    trees = []
    m = X.shape[0]
    for _ in range(n_trees):
        idx = np.random.choice(m, m, replace=True)
        tree = build_tree(X[idx], y[idx], max_depth=max_depth, criterion=criterion, extra=extra)
        trees.append(tree)
    return trees

def predict_forest(X, trees):
    preds = []
    for x in X:
        votes = [predict_tree(x, tree) for tree in trees]
        preds.append(np.bincount(votes).argmax())
    return np.array(preds)

def manual_kfold(X, y, k=5):
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    fold_sizes = np.full(k, len(X) // k, dtype=int)
    fold_sizes[:len(X) % k] += 1
    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_idx = indices[start:stop]
        train_idx = np.concatenate((indices[:start], indices[stop:]))
        folds.append((train_idx, test_idx))
        current = stop
    return folds

def main():
    data = load_wine()
    X, y = data.data, data.target
    n_trees_list = [5, 10, 20]
    depths = [3, 5, None]
    criteria = ["gini", "entropy"]
    results = {"RF": [], "ET": []}
    for clf in ["RF", "ET"]:
        for n_trees in n_trees_list:
            for d in depths:
                for crit in criteria:
                    acc_scores, f1_scores = [], []
                    for train_idx, test_idx in manual_kfold(X, y, k=5):
                        X_train, y_train = X[train_idx], y[train_idx]
                        X_test, y_test = X[test_idx], y[test_idx]
                        trees = fit_forest(X_train, y_train, n_trees=n_trees, max_depth=d,
                                           criterion=crit, extra=(clf == "ET"))
                        y_pred = predict_forest(X_test, trees)
                        acc_scores.append(accuracy_score(y_test, y_pred))
                        f1_scores.append(f1_score(y_test, y_pred, average="macro"))
                    results[clf].append((n_trees, d, crit,
                                         np.mean(acc_scores), np.mean(f1_scores)))
    for clf in results:
        print(f"\n{clf} Results:")
        for r in results[clf]:
            print(f"Trees={r[0]}, Depth={r[1]}, Criterion={r[2]}, "
                  f"Accuracy={r[3]:.3f}, F1={r[4]:.3f}")
    labels, rf_acc, et_acc = [], [], []
    for r in results["RF"]:
        labels.append(f"T{r[0]}-D{r[1]}-{r[2]}")
        rf_acc.append(r[3])
    for r in results["ET"]:
        et_acc.append(r[3])
    x = np.arange(len(labels))
    plt.figure(figsize=(12, 6))
    plt.plot(x, rf_acc, marker='o', label="Random Forest")
    plt.plot(x, et_acc, marker='s', label="Extra Trees")
    plt.xticks(x, labels, rotation=45)
    plt.ylabel("Accuracy")
    plt.title("Random Forest vs Extra Trees (Wine dataset)")
    plt.legend()
    plt.tight_layout()
    plt.show()

main()
