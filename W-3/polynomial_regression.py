import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing


def load_data():
    """Load California housing dataset and return features and target."""
    data = fetch_california_housing()
    return data.data, data.target, data.feature_names


def create_poly_features(x, degree):
    """Generate polynomial features up to the given degree."""
    x = x.reshape(-1, 1)
    X_poly = np.ones((x.shape[0], degree + 1))
    for i in range(1, degree + 1):
        X_poly[:, i] = x[:, 0] ** i
    return X_poly


def normal_equation(X, y):
    """Compute coefficients using the Normal Equation."""
    return np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))


def predict(X, beta):
    """Predict target values given features and coefficients."""
    return np.dot(X, beta)


def mse(y_true, y_pred):
    """Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)


def split_data(X, y):
    """Split data into train, validation, and test sets (60-20-20)."""
    m = len(y)
    train_end = int(0.6 * m)
    val_end = int(0.8 * m)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    return X_train, y_train, X_val, y_val, X_test, y_test


def run_regression(X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test, degree):
    """Train polynomial regression of given degree and return results."""
    # Create polynomial features
    X_train_poly = create_poly_features(X_train_raw, degree)
    X_val_poly = create_poly_features(X_val_raw, degree)
    X_test_poly = create_poly_features(X_test_raw, degree)

    # Train
    beta = normal_equation(X_train_poly, y_train)

    # Predictions
    y_train_pred = predict(X_train_poly, beta)
    y_val_pred = predict(X_val_poly, beta)
    y_test_pred = predict(X_test_poly, beta)

    # Metrics
    results = {
        "beta": beta,
        "train_mse": mse(y_train, y_train_pred),
        "val_mse": mse(y_val, y_val_pred),
        "test_mse": mse(y_test, y_test_pred),
    }

    return results


def plot_results(X_raw, y, feature_name, beta_lin, beta_poly, d):
    """Plot raw data, linear fit, and polynomial fit."""
    plt.scatter(X_raw, y, color='black', alpha=0.3, label='Raw Data')

    x_sorted = np.linspace(min(X_raw), max(X_raw), 600)

    # Linear fit
    X_sorted_lin = create_poly_features(x_sorted, 1)
    y_curve_lin = predict(X_sorted_lin, beta_lin)
    plt.plot(x_sorted, y_curve_lin, color='blue', linewidth=2, label='Linear Fit')

    # Polynomial fit
    X_sorted_poly = create_poly_features(x_sorted, d)
    y_curve_poly = predict(X_sorted_poly, beta_poly)
    plt.plot(x_sorted, y_curve_poly, color='red', linewidth=2, label=f'Polynomial Fit (d={d})')

    plt.xlabel(feature_name)
    plt.ylabel("House Value")
    plt.title("Linear vs Polynomial Regression")
    plt.legend()
    plt.show()


def main():
    # Load data
    X, y, feature_names = load_data()

    # Show available features
    print("\nAvailable features:")
    for i, name in enumerate(feature_names):
        print(f"{i}: {name}")
    feature_index = int(input("\nEnter the feature index you want to use: "))
    d = int(input("Enter polynomial degree: "))

    # Extract selected feature
    X_raw = X[:, feature_index]

    # Split dataset
    X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test = split_data(X_raw, y)

    # Run linear regression
    lin_results = run_regression(X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test, 1)

    # Run polynomial regression
    poly_results = run_regression(X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test, d)

    # --------- Print Results ---------
    print("\n Linear Regression:")
    print("Coefficients (beta):", lin_results["beta"])
    print("Train MSE:", lin_results["train_mse"])
    print("Validation MSE:", lin_results["val_mse"])
    print("Test MSE:", lin_results["test_mse"])

    print(f"\n Polynomial Regression (degree = {d})")
    print("Coefficients (beta):", poly_results["beta"])
    print("Train MSE:", poly_results["train_mse"])
    print("Validation MSE:", poly_results["val_mse"])
    print("Test MSE:", poly_results["test_mse"])

    # Plot results
    plot_results(X_raw, y, feature_names[feature_index],
                 lin_results["beta"], poly_results["beta"], d)


if __name__ == "__main__":
    main()
