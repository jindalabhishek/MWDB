import numpy as np
import random
from sklearn import preprocessing
import math
from matplotlib import pyplot as plt


def VA_Approx(X, b):
    n_samples = len(X)
    n_features = len(X[0])
    n_partitions = 2 ** b
    p = np.zeros((n_features, n_partitions+1))  # Partition points of X
    a = np.zeros((n_samples, n_features), dtype=int)  # Approx of objects
    feature_max = np.amax(X, axis=0)  # Max on every feature
    feature_min = np.amin(X, axis=0)  # Min on every feature
    bps = n_features * b  # bits per sample
    total_bits = bps * n_samples
    VA = 0

    # Find partition points
    num_samples_to_each_partition = math.floor(n_samples / n_partitions)
    X = np.array(X)
    # regions
    for i in range(0, n_features):
        image_in_dimension = np.array(X[:, i])
        sorted_indexes = np.argsort(image_in_dimension)
        sorted_indexes_chunks = np.array_split(sorted_indexes, n_partitions)
        partition_number = -1
        print(sorted_indexes_chunks)
        for k in range(0, len(sorted_indexes_chunks)):
            sorted_indexes_chunk = sorted_indexes_chunks[k]
            partition_number += 1
            if k == 0:
                p[i][partition_number] = X[sorted_indexes_chunk[0]][i]
            else:
                previous_chunk = sorted_indexes_chunks[k-1]
                bound = (X[previous_chunk[-1]][i] + X[sorted_indexes_chunk[0]][i])/2
                p[i][partition_number] = bound
            for j in range(0, len(sorted_indexes_chunk)):
                a[sorted_indexes_chunk[j]][i] = partition_number
        p[i][partition_number+1] = math.inf
    print(a)
    print(p)
    print("Total VA bytes = " + str(total_bits / 8))
    return a, p


def compute_dist(v_i, v_q, deg=2):
    ret = 0
    for i in range(0, len(v_i)):
        ret += abs(v_i[i] - v_q[i]) ** deg
    return ret ** (float(deg) ** -1.)


def compute_bound(a, p, q):
    # a = array of approximation
    # p = partition bounds
    # q = query point

    ret = 0
    n_features = len(a)
    a_q = np.zeros(n_features)

    for i in range(0, n_features):  # Find approximation of query
        for j in range(0, len(p[0])):
            if (q[i] < p[i][j]):
                a_q[i] = j - 1
                break

    for i in range(0, n_features):
        if (a[i] < a_q[i]):
            ret += (q[i] - p[i][a[i] + 1]) ** 2
        elif (a[i] > a_q[i]):
            ret += (p[i][a[i]] - q[i]) ** 2
    return ret ** (0.5)


def SortOnDst(dst, ans):
    # dst = list of distances of nearest k objects
    # ans = indices in dataset of nearest k objects

    k = len(ans)
    ret_ans = np.zeros(k)
    ret_dst = np.zeros(k)

    ind = np.argsort(dst)
    for i in range(0, k):
        ret_ans[i] = ans[ind[i]]
        ret_dst[i] = dst[ind[i]]
    return ret_dst, ret_ans


def VA_SSA(X, a, p, q, k):
    # X = dataset
    # a = array of approximation over dataset
    # p = partition bounds
    # q = query point
    # k = nearest points

    n_samples = len(X)
    d_furthest = math.inf
    ans = np.zeros(k, dtype=int)
    dst = np.array([math.inf] * k)

    # Stats
    objects_visited = []
    n_visited = 0
    buckets_visited = []
    n_buckets = 0

    for i in range(0, n_samples):
        l_i = compute_bound(a[i], p, q)
        d_i = compute_dist(X[i], q)

        if (l_i < dst[k - 1] and d_i < dst[k - 1]):
            dst[k - 1] = d_i
            ans[k - 1] = i
            dst, ans = SortOnDst(dst, ans)
    return ans


#############################
# Sample driver

# X1 = np.array([[1, 3], [2, 2], [4, 3], [6, 1], [5, 4]])
# X2 = np.array([[13, 6], [15, 8], [16, 7], [12, 9], [17, 10]])
# X3 = np.array([[2, 7], [6, 10], [3, 8], [1, 9], [7, 8]])
# X4 = np.array([[12, 3], [13, 2], [16, 4], [15, 1], [11, 3]])
# X = np.concatenate((X1, X2, X3, X4), axis=0)
# print(X)
# print('Going to call')
# a, p = VA_Approx(X, 4)
# query = [9, 6]
# ind = VA_SSA(X, a, p, query, 7)
# kNN = np.array([X[int(i)] for i in ind])
# print("7 nearest neighbors (sorted):\n" + str(kNN))
#
# plt.scatter(X[:, 0], X[:, 1], color='blue', label='dataset')
# plt.scatter(query[0], query[1], color='orange', label='Query')
# plt.scatter(kNN[:, 0], kNN[:, 1], color='red', label='7 NN to Q', marker='*')
# plt.legend()
# plt.show()
