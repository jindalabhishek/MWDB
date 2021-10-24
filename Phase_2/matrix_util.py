import numpy as np
from scipy import linalg


# Return a numpy array
def multiply_matrices(a, b):
    c = np.array(a)
    d = np.array(b)
    return np.matmul(c, d)


def inverse_matrix(a):
    b = np.array(a)
    return np.array(linalg.inv(b))


def transpose_matrix(a):
    b = np.array(a)
    return b.transpose()
