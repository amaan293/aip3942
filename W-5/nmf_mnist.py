import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist #type: ignore

def load_data(n_samples=2000):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    X = x_train.reshape(x_train.shape[0], -1).astype(np.float32)
    X /= 255.0
    return X[:n_samples]


def nmf(X, n_components, max_iter=200, tol=1e-4):
    n_samples, n_features = X.shape
    W = np.abs(np.random.rand(n_samples, n_components))
    H = np.abs(np.random.rand(n_components, n_features))
    prev_error = None

    for iteration in range(max_iter):
        WH = np.dot(W, H) + 1e-9
        H *= np.dot(W.T, X) / (np.dot(W.T, WH) + 1e-9)
        WH = np.dot(W, H) + 1e-9
        W *= np.dot(X, H.T) / (np.dot(WH, H.T) + 1e-9)
        error = np.linalg.norm(X - np.dot(W, H), 'fro')
        if prev_error is not None and abs(prev_error - error) < tol:
            break
        prev_error = error

    return W, H, error

def compute_W_for_X(X_new, H, max_iter=100, tol=1e-4):
    n_samples, n_features = X_new.shape
    n_components = H.shape[0]
    W = np.abs(np.random.rand(n_samples, n_components))
    prev_error = None

    for iteration in range(max_iter):
        WH = np.dot(W, H) + 1e-9
        W *= np.dot(X_new, H.T) / (np.dot(WH, H.T) + 1e-9)
        error = np.linalg.norm(X_new - np.dot(W, H), 'fro')
        if prev_error is not None and abs(prev_error - error) < tol:
            break
        prev_error = error

    return W

def tune_nmf(X_train, X_val, k_list=None, max_iter=100):
    if k_list is None:
        k_list = list(range(12, 21)) + [600]  
    errors = {}

    for k in k_list:
        print(f"Running NMF for n_components={k} ...")
        # Fit on training
        _, H_train, _ = nmf(X_train, n_components=k, max_iter=max_iter)
        # Compute W for validation
        W_val = compute_W_for_X(X_val, H_train, max_iter=max_iter)
        val_error = np.linalg.norm(X_val - np.dot(W_val, H_train), 'fro')
        errors[k] = val_error

    best_k = min(errors, key=errors.get)
    return best_k, errors

def show_reconstruction(X, W, H, n=10):
    X_recon = np.dot(W, H)
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(X[i].reshape(28, 28), cmap="gray")
        plt.axis("off")
        if i == 0:
            ax.set_title("Original")

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(X_recon[i].reshape(28, 28), cmap="gray")
        plt.axis("off")
        if i == 0:
            ax.set_title("Reconstructed")
    plt.show()

def plot_errors(errors):
    plt.figure(figsize=(6, 4))
    plt.plot(list(errors.keys()), list(errors.values()), marker='o')
    plt.xlabel("n_components")
    plt.ylabel("Validation Error (Frobenius Norm)")
    plt.title("NMF Hyperparameter Tuning")
    plt.show()


def main():
    X = load_data(n_samples=2000)

    # Train/Validation/Test split (70/15/15)
    n_samples = X.shape[0]
    idx = np.random.permutation(n_samples)
    train_end = int(0.7 * n_samples)
    val_end = int(0.85 * n_samples)
    X_train = X[idx[:train_end]]
    X_val = X[idx[train_end:val_end]]
    X_test = X[idx[val_end:]]

    # Tune NMF
    k_list = list(range(12, 23)) + [600]
    best_k, errors = tune_nmf(X_train, X_val, k_list=k_list, max_iter=100)
    
    # Print validation errors for all n_components
    print("\nValidation Errors for all n_components:")
    for k in sorted(errors.keys()):
        print(f"n_components={k:3d} --> Validation Error: {errors[k]:.4f}")

    print("\nBest n_components:", best_k)
    print("Validation error for n_components=600:", errors[600])

    # Train final model on train+val
    X_final_train = np.vstack([X_train, X_val])
    W_final, H_final, final_error = nmf(X_final_train, n_components=best_k, max_iter=200)
    print("Final reconstruction error on train+val:", final_error)

    # Compute W for test set
    W_test = compute_W_for_X(X_test, H_final, max_iter=200)
    test_error = np.linalg.norm(X_test - np.dot(W_test, H_final), 'fro')
    print("Test reconstruction error:", test_error)

    # Show reconstruction on test set
    show_reconstruction(X_test, W_test, H_final, n=10)

    # Plot validation error curve
    plot_errors(errors)


main()
