import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# Load dataset
data = fetch_california_housing()
X = data.data
y = data.target
feature_names = data.feature_names

# Ask user for polynomial degree
d = int(input("Enter polynomial degree (e.g., 3): "))

# Function to create polynomial features without sklearn
def create_poly_features(x, degree):
    x = x.reshape(-1, 1)
    X_poly = np.ones((x.shape[0], degree + 1))  # bias term
    for i in range(1, degree + 1):
        X_poly[:, i] = x[:, 0] ** i
    return X_poly

# Normal equation solver for beta
def normal_equation(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y

# MSE calculation
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Split dataset into train(60%), val(20%), test(20%) with shuffle
m = len(y)
indices = np.arange(m)
np.random.seed(42)
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

train_end = int(0.6 * m)
val_end = int(0.8 * m)

X_train = X[:train_end]
y_train = y[:train_end]

X_val = X[train_end:val_end]
y_val = y[train_end:val_end]

X_test = X[val_end:]
y_test = y[val_end:]

num_features = X.shape[1]
cols = 3
rows = int(np.ceil(num_features / cols))

plt.figure(figsize=(15, 12))

for i in range(num_features):
    # Extract single feature data for all splits
    X_train_f = X_train[:, i]
    X_val_f = X_val[:, i]
    X_test_f = X_test[:, i]

    # Create polynomial features (linear and chosen degree)
    X_train_lin = create_poly_features(X_train_f, 1)
    X_val_lin = create_poly_features(X_val_f, 1)
    X_test_lin = create_poly_features(X_test_f, 1)

    X_train_poly = create_poly_features(X_train_f, d)
    X_val_poly = create_poly_features(X_val_f, d)
    X_test_poly = create_poly_features(X_test_f, d)

    # Fit linear regression
    beta_lin = normal_equation(X_train_lin, y_train)
    # Fit polynomial regression
    beta_poly = normal_equation(X_train_poly, y_train)

    # Predictions
    y_train_pred_lin = X_train_lin @ beta_lin
    y_val_pred_lin = X_val_lin @ beta_lin
    y_test_pred_lin = X_test_lin @ beta_lin

    y_train_pred_poly = X_train_poly @ beta_poly
    y_val_pred_poly = X_val_poly @ beta_poly
    y_test_pred_poly = X_test_poly @ beta_poly

    # Print results for this feature
    print(f"\nFeature: {feature_names[i]}")
    print(" Linear Regression Coefficients:", beta_lin)
    print("  Train MSE:", mse(y_train, y_train_pred_lin))
    print("  Validation MSE:", mse(y_val, y_val_pred_lin))
    print("  Test MSE:", mse(y_test, y_test_pred_lin))
    print(f" Polynomial Regression (deg={d}) Coefficients:", beta_poly)
    print("  Train MSE:", mse(y_train, y_train_pred_poly))
    print("  Validation MSE:", mse(y_val, y_val_pred_poly))
    print("  Test MSE:", mse(y_test, y_test_pred_poly))

    # Plotting
    plt.subplot(rows, cols, i+1)

    # Scatter plot all raw points (for visualization only using whole data)
    plt.scatter(X[:, i], y, color='gray', alpha=0.3, s=10, label='Raw data')

    # For smooth plotting, create sorted x values over full feature range
    x_sorted = np.linspace(X[:, i].min(), X[:, i].max(), 500)
    X_plot_lin = create_poly_features(x_sorted, 1)
    X_plot_poly = create_poly_features(x_sorted, d)

    y_plot_lin = X_plot_lin @ beta_lin
    y_plot_poly = X_plot_poly @ beta_poly

    plt.plot(x_sorted, y_plot_lin, color='blue', label='Linear Fit')
    plt.plot(x_sorted, y_plot_poly, color='red', label=f'Poly Fit (deg {d})')

    plt.title(feature_names[i])
    plt.xlabel(feature_names[i])
    plt.ylabel("House Value")
    if i == 0:
        plt.legend()

plt.tight_layout()
plt.show()
