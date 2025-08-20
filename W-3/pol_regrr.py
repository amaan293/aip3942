import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing


def create_poly_features(x, degree):
    x = x.reshape(-1, 1)
    X_poly = np.ones((x.shape[0], degree + 1))
    for i in range(1, degree + 1):
        X_poly[:, i] = x[:, 0] ** i
    return X_poly


def normal_equation(X, y):
    return np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def train_val_test_split(X, y, train_ratio=0.6, val_ratio=0.2, seed=42):
    m = len(y)
    indices = np.arange(m)
    np.random.seed(seed)
    np.random.shuffle(indices)

    X, y = X[indices], y[indices]
    train_end = int(train_ratio * m)
    val_end = int((train_ratio + val_ratio) * m)

    return (X[:train_end], y[:train_end],
            X[train_end:val_end], y[train_end:val_end],
            X[val_end:], y[val_end:])


def evaluate_feature(X_train, y_train, X_val, y_val, X_test, y_test, feature, degree, feature_name):
    X_train_f, X_val_f, X_test_f = X_train[:, feature], X_val[:, feature], X_test[:, feature]

    X_train_lin = create_poly_features(X_train_f, 1)
    X_val_lin = create_poly_features(X_val_f, 1)
    X_test_lin = create_poly_features(X_test_f, 1)

    X_train_poly = create_poly_features(X_train_f, degree)
    X_val_poly = create_poly_features(X_val_f, degree)
    X_test_poly = create_poly_features(X_test_f, degree)

    beta_lin = normal_equation(X_train_lin, y_train)
    beta_poly = normal_equation(X_train_poly, y_train)

    y_train_pred_lin = np.dot(X_train_lin, beta_lin)
    y_val_pred_lin = np.dot(X_val_lin, beta_lin)
    y_test_pred_lin = np.dot(X_test_lin, beta_lin)

    y_train_pred_poly = np.dot(X_train_poly, beta_poly)
    y_val_pred_poly = np.dot(X_val_poly, beta_poly)
    y_test_pred_poly = np.dot(X_test_poly, beta_poly)

    print(f"\nFeature: {feature_name}")
    print(" Linear Regression Coefficients:", beta_lin)
    print("  Train MSE:", mse(y_train, y_train_pred_lin))
    print("  Validation MSE:", mse(y_val, y_val_pred_lin))
    print("  Test MSE:", mse(y_test, y_test_pred_lin))

    print(f" Polynomial Regression (deg={degree}) Coefficients:", beta_poly)
    print("  Train MSE:", mse(y_train, y_train_pred_poly))
    print("  Validation MSE:", mse(y_val, y_val_pred_poly))
    print("  Test MSE:", mse(y_test, y_test_pred_poly))

    return beta_lin, beta_poly


def plot_feature_fit(X, y, feature, degree, beta_lin, beta_poly, feature_name, subplot_index, rows, cols):
    plt.subplot(rows, cols, subplot_index)
    plt.scatter(X[:, feature], y, color='gray', alpha=0.3, s=10, label='Raw data')

    x_sorted = np.linspace(X[:, feature].min(), X[:, feature].max(), 500)
    X_plot_lin = create_poly_features(x_sorted, 1)
    X_plot_poly = create_poly_features(x_sorted, degree)

    y_plot_lin = np.dot(X_plot_lin, beta_lin)
    y_plot_poly = np.dot(X_plot_poly, beta_poly)

    plt.plot(x_sorted, y_plot_lin, color='blue', label='Linear Fit')
    plt.plot(x_sorted, y_plot_poly, color='red', label=f'Poly Fit (deg {degree})')

    plt.title(feature_name)
    plt.xlabel(feature_name)
    plt.ylabel("House Value")

    if subplot_index == 1:
        plt.legend()


def main():
    d = int(input("Enter polynomial degree (e.g., 3): "))
    data = fetch_california_housing()
    X, y, feature_names = data.data, data.target, data.feature_names

    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y)

    num_features = X.shape[1]
    cols = 3
    rows = int(np.ceil(num_features / cols))

    plt.figure(figsize=(15, 12))

    for i in range(num_features):
        beta_lin, beta_poly = evaluate_feature(X_train, y_train, X_val, y_val, X_test, y_test, i, d, feature_names[i])
        plot_feature_fit(X, y, i, d, beta_lin, beta_poly, feature_names[i], i+1, rows, cols)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
