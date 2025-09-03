import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- Load Data ---
df = pd.read_csv(r'D:\aip3942\W-6\Iris.csv')
X = df.iloc[:, 1:5].to_numpy()
y = df.iloc[:, 5].to_numpy()

# --- Functions ---
def standardize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_std = (X - mu) / sigma
    print("Shape of standardized data X_std:", X_std.shape)
    return X_std, mu, sigma    

def covariance(X_std):
    n = X_std.shape[0]
    cov_matrix = np.dot(X_std.T, X_std) / (n - 1)   
    print("\nCovariance Matrix:\n", cov_matrix)
    return cov_matrix

def power_iteration(A, num_iter=1000, tol=1e-6):
    n = A.shape[0]
    b = np.random.rand(n)
    b = b / np.linalg.norm(b)
    
    for _ in range(num_iter):
        Ab = np.dot(A, b)
        b_next = Ab / np.linalg.norm(Ab)
        if np.linalg.norm(b - b_next) < tol:
            break
        b = b_next
    eigenvalue = np.dot(b.T, np.dot(A, b))
    return eigenvalue, b

def eigen_decomposition(A, k=None):
    n = A.shape[0]
    if k is None: 
        k = n
    eigenvalues = []
    eigenvectors = []
    A_copy = A.copy()

    for i in range(k):
        val, vec = power_iteration(A_copy)
        eigenvalues.append(val)
        eigenvectors.append(vec)
        A_copy = A_copy - val * np.outer(vec, vec)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.array(eigenvalues)[idx]
    eigenvectors = np.array(eigenvectors)[idx].T
    return eigenvalues, eigenvectors

def explained_variance(sorted_eigen_values):
    variance_ratio = sorted_eigen_values / np.sum(sorted_eigen_values)
    cumulative = np.cumsum(variance_ratio)
    print("\nExplained Variance Ratio:\n", variance_ratio)
    print("Cumulative Variance:\n", cumulative)
    return variance_ratio, cumulative

def projection(X_std, sorted_eigen_vectors):
    W_matrix = sorted_eigen_vectors[:, :2]
    print("\nProjection Matrix W:\n", W_matrix)

    X_pca = np.dot(X_std, W_matrix)
    print("\nReformed Matrix shape:", X_pca.shape)
    print("\nFirst five rows of X_pca:\n", X_pca[:5])

    X_recon_approx = np.dot(X_pca, W_matrix.T)
    diff = X_std - X_recon_approx
    mse = np.mean(np.square(diff))
    print(" Reconstruction MSE:", mse)
    return W_matrix, X_pca, mse

def vizualise_pca(X_pca, y, title="PCA on Iris dataset"):
    classes = np.unique(y)
    colors = ['r', 'g', 'b']
    
    plt.figure(figsize=(8,6))
    for target, color in zip(classes, colors):
        plt.scatter(X_pca[y==target,0],
                    X_pca[y==target,1],
                    c=color, label=target, alpha=0.7, edgecolors='k')
                    
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# --- Standard PCA on all features ---
X_std, mu, sigma = standardize(X)
cov_matrix = covariance(X_std)
eigen_values, eigen_vectors = eigen_decomposition(cov_matrix, k=4)

print("\n Eigenvalues:\n", eigen_values)
print("\n Eigenvectors: \n", eigen_vectors)  # one column is one eigenvector

variance_ratio, cumulative = explained_variance(eigen_values)
W, X_pca, mse = projection(X_std, eigen_vectors)
vizualise_pca(X_pca, y)

# --- NEW: PCA on two selected features, plot only PC1 ---
# Select any two features (e.g., SepalLength & SepalWidth)
feature_indices = [0, 1]  # change this if you want different features
X_2d = X_std[:, feature_indices]
selected_feature_names = df.columns[1:5][feature_indices].to_list()

# Covariance and eigen decomposition
cov_2d = covariance(X_2d)
eig_vals_2d, eig_vecs_2d = eigen_decomposition(cov_2d, k=2)

# Take the first principal component (PC1)
pc1 = eig_vecs_2d[:, 0]

# Plot 2D data + PC1
plt.figure(figsize=(6,6))
plt.scatter(X_2d[:,0], X_2d[:,1], alpha=0.6, edgecolors='k')
mean_2d = np.mean(X_2d, axis=0)
plt.arrow(mean_2d[0], mean_2d[1],
          pc1[0]*2, pc1[1]*2,  # scale factor for visibility
          color='r', head_width=0.05, linewidth=2, label='PC1')

plt.xlabel(selected_feature_names[0])
plt.ylabel(selected_feature_names[1])
plt.title(f"PC1 in {selected_feature_names[0]} vs {selected_feature_names[1]} Feature Space")
plt.legend()
plt.grid(True)
plt.show()
