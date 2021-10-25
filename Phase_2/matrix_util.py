import numpy as np
from scipy import linalg


# Return a numpy array
def make_matrix(a):
    return np.array([np.array(x) for x in a])


def multiply_matrices(a, b):
    c = np.array(a)
    d = np.array(b)
    return np.matmul(c, d)


def inverse_matrix(a):
    b = np.array(a)
    return linalg.inv(b)


def transpose_matrix(a):
    return np.array(a).transpose()
