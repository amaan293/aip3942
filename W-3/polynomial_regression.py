import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing()
feature_names = data.feature_names

print("\nAvailable features:")
for i, name in enumerate(feature_names):
    print(f"{i}: {name}")
feature_index = int(input("\nEnter the feature index you want to use: "))

#X= data.data() #sklearn method to extract all the features
X_raw = data.data[:, feature_index]  
y = data.target
print(data.target[:5])

def create_poly_features(x, degree):
    x = x.reshape(-1, 1)
    X_poly = np.ones((x.shape[0], degree + 1))  
    for i in range(1, degree + 1):
        X_poly[:, i] = x[:, 0] ** i
    return X_poly

def normal_equation(X, y):
    return np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))

def predict(X, beta):
    return np.dot(X, beta)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

m = len(y)
train_end = int(0.6 * m)
val_end = int(0.8 * m)

X_train_raw, y_train = X_raw[:train_end], y[:train_end]
X_val_raw, y_val = X_raw[train_end:val_end], y[train_end:val_end]
X_test_raw, y_test = X_raw[val_end:], y[val_end:]

X_train_lin = create_poly_features(X_train_raw, 1)
X_val_lin = create_poly_features(X_val_raw, 1)
X_test_lin = create_poly_features(X_test_raw, 1)

beta_lin = normal_equation(X_train_lin, y_train)

y_train_pred_lin = predict(X_train_lin, beta_lin)
y_val_pred_lin = predict(X_val_lin, beta_lin)
y_test_pred_lin = predict(X_test_lin, beta_lin)

d = int(input("Enter polynomial degree: "))
X_train_poly = create_poly_features(X_train_raw, d)
X_val_poly = create_poly_features(X_val_raw, d)
X_test_poly = create_poly_features(X_test_raw, d)

beta_poly = normal_equation(X_train_poly, y_train)

y_train_pred_poly = predict(X_train_poly, beta_poly)
y_val_pred_poly = predict(X_val_poly, beta_poly)
y_test_pred_poly = predict(X_test_poly, beta_poly)

# --------- Print Results ---------
print("\n Linear Regression:")
print("Coefficients (beta):", beta_lin)
print("Train MSE:", mse(y_train, y_train_pred_lin))
print("Validation MSE:", mse(y_val, y_val_pred_lin))
print("Test MSE:", mse(y_test, y_test_pred_lin))

print(f"\n Polynomial Regression (degree = {d})")
print("Coefficients (beta):", beta_poly)
print("Train MSE:", mse(y_train, y_train_pred_poly))
print("Validation MSE:", mse(y_val, y_val_pred_poly))
print("Test MSE:", mse(y_test, y_test_pred_poly))

plt.scatter(X_raw, y, color='black', alpha=0.3, label='Raw Data')

x_sorted = np.linspace(min(X_raw), max(X_raw), 600)

X_sorted_lin = create_poly_features(x_sorted, 1)
y_curve_lin = predict(X_sorted_lin, beta_lin)
plt.plot(x_sorted, y_curve_lin, color='blue', linewidth=2, label='Linear Fit')

X_sorted_poly = create_poly_features(x_sorted, d)
y_curve_poly = predict(X_sorted_poly, beta_poly)
plt.plot(x_sorted, y_curve_poly, color='red', linewidth=2, label=f'Polynomial fit')

plt.xlabel(feature_names[feature_index])
plt.ylabel("House Value")
plt.title("Linear vs Polynomial Regression")
plt.legend()
plt.show()
