import numpy as np
from mnist_preprocess import load_mnist, preprocess, stratified_split


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision(y_true, y_pred, average="macro"):
    classes = np.unique(y_true)
    precisions = []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        precisions.append(tp / (tp + fp + 1e-9))
    return np.mean(precisions) if average == "macro" else precisions

def recall(y_true, y_pred, average="macro"):
    classes = np.unique(y_true)
    recalls = []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fn = np.sum((y_pred != c) & (y_true == c))
        recalls.append(tp / (tp + fn + 1e-9))
    return np.mean(recalls) if average == "macro" else recalls

def f1_score(y_true, y_pred, average="macro"):
    p = precision(y_true, y_pred, average=None)
    r = recall(y_true, y_pred, average=None)
    f1 = [2 * pi * ri / (pi + ri + 1e-9) for pi, ri in zip(p, r)]
    return np.mean(f1) if average == "macro" else f1


def train_linear_svm(X, y, C=1.0, lr=1e-3, epochs=5):
    N, D = X.shape
    classes = np.unique(y)
    W = np.zeros((len(classes), D))
    b = np.zeros(len(classes))

    for epoch in range(epochs):
        for i in range(N):
            xi = X[i]
            yi = y[i]
            for ci, c in enumerate(classes):
                yi_c = 1 if yi == c else -1
                score = np.dot(W[ci], xi) + b[ci]
                margin = yi_c * score
                if margin < 1:
                    W[ci] = (1 - lr) * W[ci] + lr * C * yi_c * xi
                    b[ci] += lr * C * yi_c
                else:
                    W[ci] = (1 - lr) * W[ci]
        print(f"Linear SVM Epoch {epoch+1}/{epochs} completed")

    return W, b, classes

def predict_linear_svm(X, W, b, classes):
    scores = X @ W.T + b
    preds = np.argmax(scores, axis=1)
    return classes[preds]


def polynomial_kernel(X1, X2, degree=2):
    return (X1 @ X2.T + 1) ** degree

def train_quadratic_svm(X, y, C=1.0, lr=1e-4, epochs=20, degree=2):
    N = X.shape[0]
    classes = np.unique(y)
    alphas = np.zeros((len(classes), N))
    b = np.zeros(len(classes))
    K = polynomial_kernel(X, X, degree=degree)

    for epoch in range(epochs):
        for i in range(N):
            xi_label = y[i]
            for ci, c in enumerate(classes):
                yi_c = 1 if xi_label == c else -1
                decision = np.sum(alphas[ci] * (np.where(y == c, 1, -1)) * K[:, i])
                grad = 1 - yi_c * decision
                alphas[ci, i] = np.clip(alphas[ci, i] + lr * grad, 0, C)
        print(f"Quadratic SVM Epoch {epoch+1}/{epochs} completed")

    for ci, c in enumerate(classes):
        yi_c = np.where(y == c, 1, -1)
        support_idx = np.where((alphas[ci] > 1e-5) & (alphas[ci] < C))[0]
        b[ci] = np.mean(yi_c[support_idx] - np.sum((alphas[ci] * yi_c)[:, None] * K[:, support_idx], axis=0)) if len(support_idx) > 0 else 0.0

    return alphas, b, classes, K, degree, X, y

def predict_quadratic_svm(X_test, alphas, b, classes, degree, X_train, y_train):
    K_test = polynomial_kernel(X_test, X_train, degree=degree)
    scores = np.zeros((X_test.shape[0], len(classes)))
    for ci, c in enumerate(classes):
        yi_c = np.where(y_train == c, 1, -1)
        scores[:, ci] = K_test @ (alphas[ci] * yi_c) + b[ci]
    preds = np.argmax(scores, axis=1)
    return classes[preds]
