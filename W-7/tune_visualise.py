import numpy as np
import matplotlib.pyplot as plt
from mnist_preprocess import load_mnist, preprocess, stratified_split
from svm_mnist import train_linear_svm, predict_linear_svm, train_quadratic_svm, predict_quadratic_svm

def pca_2d(X):
    X_centered = X - np.mean(X, axis=0)
    cov = np.cov(X_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    top2 = eigvecs[:, -2:]
    return X_centered @ top2

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def tune_linear_svm(X_tr, y_tr, X_val, y_val, C_values, lr=1e-3, epochs=5):
    results = []
    for C in C_values:
        W, b, classes = train_linear_svm(X_tr, y_tr, C=C, lr=lr, epochs=epochs)
        y_val_pred = predict_linear_svm(X_val, W, b, classes)
        acc = accuracy(y_val, y_val_pred)
        results.append((C, acc))
        print(f"Linear SVM C={C}, Val Accuracy={acc:.4f}")
    return results

def tune_quadratic_svm(X_tr, y_tr, X_val, y_val, C_values, degree_values, lr=1e-4, epochs=20):
    results = []
    for degree in degree_values:
        for C in C_values:
            alphas, b, classes, K_train, deg, X_tr_quad, y_tr_quad = train_quadratic_svm(
                X_tr, y_tr, C=C, lr=lr, epochs=epochs, degree=degree
            )
            y_val_pred = predict_quadratic_svm(X_val, alphas, b, classes, deg, X_tr_quad, y_tr_quad)
            results.append((degree, C, y_val_pred))
            acc = accuracy(y_val, y_val_pred)
            print(f"Quadratic SVM degree={degree}, C={C}, Val Accuracy={acc:.4f}")
    return results

def plot_decision_boundary_2d(X_tr, y_tr, model_predict, model_params, title):
    X_2d = pca_2d(X_tr)
    grid_x, grid_y = np.meshgrid(
        np.linspace(X_2d[:,0].min()-0.1, X_2d[:,0].max()+0.1, 100),
        np.linspace(X_2d[:,1].min()-0.1, X_2d[:,1].max()+0.1, 100)
    )
    grid = np.c_[grid_x.ravel(), grid_y.ravel()]

    mean_X = np.mean(X_tr, axis=0)
    X_centered = X_tr - mean_X
    cov = np.cov(X_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    top2 = eigvecs[:, -2:]
    grid_orig = grid @ top2.T + mean_X

    Z = model_predict(grid_orig, *model_params)
    Z = Z.reshape(grid_x.shape)

    plt.figure(figsize=(12, 5))
    plt.contourf(grid_x, grid_y, Z, alpha=0.3, cmap=plt.cm.tab10)
    for i in np.unique(y_tr):
        plt.scatter(X_2d[y_tr==i,0], X_2d[y_tr==i,1], label=str(i), s=20)
    plt.title(title)
    plt.legend()
    plt.show()


