import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv(r'D:\aip3942\W-6\Iris.csv')

X = df.iloc[:,1:5].to_numpy()
y = df.iloc[:,5].to_numpy()

def standardize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_std = (X - mu) / sigma
    print("Shape of X_std:", X_std.shape)
    return X_std, mu, sigma    

def covariance(X_std):
    cov_matrix = np.cov(X_std.T)
    print("The shape of the Covariance Matrix is: ", cov_matrix.shape)
    return cov_matrix

def eigen_compute(cov_matrix):
    eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)
    indices = np.argsort(eigen_values)[::-1]
    sorted_eigen_values = eigen_values[indices]
    sorted_eigen_vectors = eigen_vectors[:, indices]
    return eigen_values, eigen_vectors, sorted_eigen_values, sorted_eigen_vectors

def explained_variance(sorted_eigen_values):
    variance_ratio = sorted_eigen_values / np.sum(sorted_eigen_values)
    cumulative = np.cumsum(variance_ratio)
    return variance_ratio , cumulative

def projection(X_std, sorted_eigen_vectors):
    W_matrix = sorted_eigen_vectors[:,:2]
    print("Projection Matrix W: \n", W_matrix)

    X_pca = X_std.dot(W_matrix)
    print("Reformed Matrix shape:", X_pca.shape)
    print("First five rows of the X_pca are: \n", X_pca[:5])

    X_recon_approx = X_pca.dot(W_matrix.T)
    diff = X_std - X_recon_approx
    mse = np.mean(np.square(diff))
    print("Reconstruction MSE:", mse)
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


X_std, mu, sigma = standardize(X)
cov_matrix = covariance(X_std)
eigen_values, eigen_vectors, sorted_eigen_values, sorted_eigen_vectors = eigen_compute(cov_matrix)
variance_ratio, cumulative = explained_variance(sorted_eigen_values)
W, X_pca, mse = projection(X_std, sorted_eigen_vectors)
vizualise_pca(X_pca, y)
