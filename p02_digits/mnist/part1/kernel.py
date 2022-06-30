import numpy as np

### Functions for you to fill in ###



def polynomial_kernel(X, Y, c, p):
    """
        Compute the polynomial kernel between two matrices X and Y::
            K(x, y) = (<x, y> + c)^p
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            c - a coefficient to trade off high-order and low-order terms (scalar)
            p - the degree of the polynomial kernel

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    return (X @ np.transpose(Y) + c)**p




def rbf_kernel(X, Y, gamma):
    """
        Compute the Gaussian RBF kernel between two matrices X and Y::
            K(x, y) = exp(-gamma ||x-y||^2)
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            gamma - the gamma parameter of gaussian function (scalar)

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    Z = X - Y
    return np.exp(-gamma * np.dot(Z, Z.transpose()))


def rbf_kernel_solution(X, Y, gamma):
    """
        Compute the Gaussian RBF kernel between two matrices X and Y::
            K(x, y) = exp(-gamma ||x-y||^2)
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            gamma - the gamma parameter of gaussian function (scalar)

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    XTX = np.mat([np.dot(row, row) for row in X]).T
    YTY = np.mat([np.dot(row, row) for row in Y]).T
    XTX_matrix = np.repeat(XTX, Y.shape[0], axis=1)
    YTY_matrix = np.repeat(YTY, X.shape[0], axis=1).T
    K = np.asarray((XTX_matrix + YTY_matrix - 2 * (X @ Y.T)), dtype='float64')
    K *= -gamma
    return np.exp(K, K)

X = np.array([[1,2,3], [4,5,6], [1,2,3]])
Y = np.array([[1,2,1], [2,1,2], [1,2,1]])
gamma = 1

# print('X-Y:\n', X-Y)
# print('my solution:\n', rbf_kernel(X,Y,gamma))
print('their solution:\n', rbf_kernel_solution(X,Y,gamma))
