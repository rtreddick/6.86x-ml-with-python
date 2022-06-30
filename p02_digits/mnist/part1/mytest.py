import numpy as np
from scipy import sparse


def create_data(n, d, k, X_min, X_max, theta_min, theta_max, random_seed=None):

    rng = np.random.default_rng(random_seed)

    # X is n x d (observations x features)
    X = rng.integers(X_min, X_max+1, (n,d))

    # Y is n x 1 (observations)
    Y = rng.integers(1, k+1, (n,1))
    
    # theta is k x d (classes x features)
    theta = rng.integers(theta_min, theta_max+1, (k,d))

    return X, Y, theta

def compute_probabilities(X, theta, temp_parameter):
    """
    Computes, for each datapoint X[i], the probability that X[i] is labeled as j
    for j = 0, 1, ..., k-1

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        theta - (k, d) NumPy array, where row j represents the parameters of our model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)
    Returns:
        H - (k, n) NumPy array, where each entry H[j][i] is the probability that X[i] is labeled as j
    """
    H = theta @ np.transpose(X)
    H = np.exp((H - np.amax(H, 0, keepdims=True))/temp_parameter)
    H = H / np.sum(H, 0, keepdims=True)

    return H

def compute_cost_function(X, Y, theta, lambda_factor, temp_parameter):
    """
    Computes the total cost over every datapoint.

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns
        c - the cost value (scalar)
    """
    H = compute_probabilities(X, theta, temp_parameter)
    k, n = H.shape[0], H.shape[1]
    loss = 0
    for i in range(n):
        for j in range(k):
            if Y[i] == j:
                loss += np.log(H[j][i])
    c = -1/n * np.sum(loss) + lambda_factor/2 * np.sum(theta**2)

    return c


X, Y, theta = create_data(5, 2, 3, 1, 10, 1, 5, 2022)
lambda_factor, temp_parameter = 1, 1


# test = (Y,range(10))
# print(test)

# num_labels = 10
# num_examples = len(Y)


# M = sparse.coo_matrix(([1]*num_examples, (Y,range(num_examples))), shape=(num_labels,num_examples)).toarray()
# print(M)

# row  = np.array([0, 0, 1, 3, 1, 0, 0])
# col  = np.array([0, 2, 1, 3, 1, 0, 0])
# data = np.array([1, 1, 1, 1, 1, 1, 1])
# coo = sparse.coo_matrix((data, (row, col)), shape=(4, 4))
# # Duplicate indices are maintained until implicitly or explicitly summed
# print(np.max(coo.data))
# print(coo.toarray())

num_examples = 6
data = [1] * num_examples

Y = np.array([0,1,1,0,0,2])

# M[i][j] = 1 if y^(j) = i and 0 otherwise.
M = sparse.coo_matrix(([1]*num_examples, (Y, range(num_examples))), shape=(3,num_examples)).toarray()
print(M)
print(M[:,:2])
