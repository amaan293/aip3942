import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_csv(r'D:\aip3942\W-6\Iris.csv')

X=df.iloc[:,1:5].to_numpy()
y=df.iloc[:, 5].to_numpy()
# print(type(X))           to verify the class of the dataset
#print((X.shape))         # to get to know about the size of X
#print(len(y))            # to know about the size of y distribution over samples


mu= np.mean(X, axis=0)
sigma= np.std(X, axis=0)

X_std= (X-mu)/sigma
print("Shape of standardized data:", X_std.shape)      # verification of the shape

#print(mu, sigma)        # check for mean and std
#print(X_std[:5])


cov_matrix= np.cov(X_std.T)   # np.cov does the whole calc of solving a covariance matrix
print("The shape of the COvariance Matrix is: ", cov_matrix.shape)
print("\n Calculated Covariance Matrix is: \n" , cov_matrix)


eigen_values, eigen_vectors= np.linalg.eig(cov_matrix)

print("\n Eigen Values of the Cov Matrix are: \n", eigen_values)
print("\n Eigen Vectors of the Cov Matrix are: \n", eigen_vectors)


print("Dot Product of two Eigen Vectors: \n", np.dot(eigen_vectors[:, 0], eigen_vectors[:, 1]))


indices=np.argsort(eigen_values)[::-1] #print the descending sorted eigen_vals
sorted_eigen_values= eigen_values[indices]
sorted_eigen_vectors= eigen_vectors[: , indices]

print("Sorted eigen values are: ", sorted_eigen_values)
print("Sorted eigen vectors are: ", sorted_eigen_vectors)

variance_ratio= sorted_eigen_values/np.sum(sorted_eigen_values)
print("\nExplained variance ratio", variance_ratio )
print("Cumulative Sum: ", np.cumsum(variance_ratio))


W_matrix= sorted_eigen_vectors[:,:2]
print("Projection Matrix W: \n", W_matrix)

X_pca= X_std.dot(W_matrix)
print("Reformed Matrix shape:", X_pca.shape)
print("First five rows of the X_pca are: \n", X_pca[:5])

X_recon_approx = X_pca.dot(W_matrix.T) #reforming the X_pca to back to 4 dimensions to check whether we did the correct transformation
#print(X_recon_approx)

diff = X_std - X_recon_approx
# Mean squared reconstruction error
mse = np.mean(np.square(diff))
print("Reconstruction MSE:", mse)


def vizualise_pca(X_pca, y , title="PCA on Iris dataset"):
    
    """
    X_pca= nd_array of the transformed dataset (n_samples=150, 2)
    y= color labels of the species name
    """
    
    classes= np.unique(y)
    colors=['r', 'g', 'b']
    
    plt.figure(figsize=(8,6))
    for target, color in zip(classes, colors):
        plt.scatter(X_pca[y==target,0],
                    X_pca[y==target,1],
                    c=color, label=target, alpha=0.7, edgecolors='k')
                    
                    
    plt.xlabel="PC1"
    plt.ylabel="PC2"
    plt.title(title)
    plt.legend
    plt.grid(True)
    plt.show()

vizualise_pca(X_pca, y)    
