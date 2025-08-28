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
print(X_std.shape)      # verification of the shape

#print(mu, sigma)        # check for mean and std
#print(X_std[:5])


cov_matrix= np.cov(X_std.T)   # np.cov does the whole calc of solving a covariance matrix
print("The shape of the COvariance Matrix is: ", cov_matrix.shape)
print("\n Calculated Covariance Matrix is: \n" , cov_matrix)


eigen_values, eigen_vectors= np.linalg.eig(cov_matrix)

print("\n Eigen Values of the Cov Matrix are: \n", eigen_values)
print("\n Eigen Vectors of the Cov Matrix are: \n", eigen_vectors)


print("Dot Product of two Eigen Vectors: \n", np.dot(eigen_vectors[:, 0], eigen_vectors[:, 1]))