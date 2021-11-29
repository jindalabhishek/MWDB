import numpy as np
import random
import math
from sklearn import preprocessing
from matplotlib import pyplot as plt

random.seed(0)


## SVM
# Multi-class SVM using OVR
# Uses linear kernel function
#  Non-linear doesn't improve results in high dimensions
#   https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf
# Implemented by SMO algorithm
#   http://chubakbidpaa.com/assets/pdf/smo.pdf


def kernel(X_i, X):
    return np.inner(X_i, X)


def classifier(alpha, X_i, X, Y, b):
    return (np.inner((alpha * Y), kernel(X_i, X)) + b)


def compute_error(alpha, X, Y, b, ind):
    return classifier(alpha, X[ind], X, Y, b) - Y[ind]


def compute_bounds(alpha_i, alpha_j, Y_i, Y_j, C):
    if (Y_i != Y_j):
        L = max(0, alpha_j - alpha_i)
        H = min(C, C + alpha_j - alpha_i)
        return (L, H)
    else:
        L = max(0, alpha_i + alpha_j - C)
        H = min(C, alpha_i + alpha_j)
        return (L, H)


def compute_eta(X_i, X_j):  # Compute eta using kernel function
    return 2 * (np.inner(X_i, X_j)) - np.inner(X_i, X_i) - np.inner(X_j, X_j)
    # return 2*X_i*X_j.T - X_i*X_i.T - X_j*X_j.T


def compute_alpha_j(alpha, Y_j, E_i, E_j, eta, L, H):
    new_alpha = alpha - Y_j * (E_i - E_j) / eta

    if (new_alpha > H):
        return H
    elif (new_alpha < L):
        return L
    else:
        return new_alpha


def compute_alpha_i(alpha_i, old_alpha_j, alpha_j, Y_i, Y_j):
    return alpha_i + (Y_i * Y_j * (old_alpha_j - alpha_j))


def compute_threshold(b, C, \
                      X_i, X_j, Y_i, Y_j, E_i, E_j, \
                      alpha_i, alpha_j, old_alpha_i, old_alpha_j):
    b_1 = b - E_i - (Y_i) * (alpha_i - old_alpha_i) * np.inner(X_i, X_i) - (
                Y_j * (alpha_j - old_alpha_j) * np.inner(X_i, X_j))
    b_2 = b - E_j - (Y_i) * (alpha_i - old_alpha_i) * np.inner(X_i, X_j) - (
                Y_j * (alpha_j - old_alpha_j) * np.inner(X_j, X_j))

    if (alpha_i > 0 and alpha_i < C):
        return b_1
    elif (alpha_j > 0 and alpha_j < C):
        return b_2
    else:
        return (b_1 + b_2) / 2.


def find_j(i, max):
    j = random.randint(0, max)
    while (i == j): j = random.randint(0, max)
    return j

    # DON'T FORGET TO PRE-PROCESS BEFORE FITTING


def SMO(X, Y, C, eps=1e-3, max_iteration=10):  # PUT args into class member
    # X, Y: np array

    n_samples = X.shape[0]
    alpha = np.zeros(n_samples)
    old_alpha = np.zeros(n_samples)
    threshold = iteration = 0

    while (iteration < max_iteration):
        # print(iteration, " ", max_iteration)
        n_changed_alpha = 0
        for i in range(0, n_samples):
            E_i = compute_error(alpha, X, Y, threshold, i)
            if ((Y[i] * E_i < -eps and alpha[i] < C) or (Y[i] * E_i > eps and alpha[i] > 0)):
                j = find_j(i, n_samples - 1)
                E_j = compute_error(alpha, X, Y, threshold, j)
                old_alpha[i] = alpha[i]
                old_alpha[j] = alpha[j]
                L, H = compute_bounds(alpha[i], alpha[j], Y[i], Y[j], C)
                if (L == H): continue
                eta = compute_eta(X[i], X[j])
                if (eta >= 0): continue
                alpha[j] = compute_alpha_j(alpha[j], Y[j], E_i, E_j, eta, L, H)
                if (abs(float(alpha[j] - old_alpha[j])) < 0.000005): continue
                alpha[i] = compute_alpha_i(alpha[i], old_alpha[j], alpha[j], Y[i], Y[j])
                threshold = compute_threshold(threshold, C, X[i], X[j], Y[i], Y[j], E_i, E_j, alpha[i], alpha[j],
                                              old_alpha[i], old_alpha[j])
                n_changed_alpha += 1
        if (n_changed_alpha == 0):
            iteration += 1
        else:
            iteration = 0
    return alpha, threshold


def binary_train(X, Y):
    # Compute support vectors (bounds) on binary classes

    X = np.array(X)
    Y = np.array(Y)
    classes = np.unique(Y)
    n_features = X.shape[1]
    n_samples = X.shape[0]

    if (len(classes) != 2):
        print("FATAL: binary train has more than 2 classes - exiting")
        exit()

    alpha, threshold = SMO(X, Y, 0.5)

    W = np.zeros(n_features)
    for i in range(0, n_samples):
        W += alpha[i] * Y[i] * X[i]

    return alpha, threshold, W


def multiclass_train(X, Y):
    # Compute bounds on multiple classes with OVR approach
    # Return a list of n_classes, each with alpha, b and W

    X = np.array(X)
    Y = np.array(Y)
    classes = np.unique(Y)
    n_classes = len(classes)
    n_samples = len(X)

    if n_classes <= 2:
        print("FATAL: multiclass train has fewer than 2 classes - exiting")
        exit()

    ret = []
    for class_i in range(0, n_classes):
        # Binarize Y
        Y_binary = np.ones(n_samples)
        for j in range(0, n_samples):
            if Y[j] != classes[class_i]:
                Y_binary[j] = -1

        alpha, threshold, W = binary_train(X, Y_binary)
        # ret.append((alpha,threshold, W, Y_binary))
        ret.append({'alpha': alpha, 'threshold': threshold, 'W': W, 'Y_binary': Y_binary})
    return {'classifiers': ret, 'classes': classes}


def binary_classifier(alpha, threshold, W, X_i, X, Y, plot=False):
    # Classify datapoints from computed support vectors
    # https://stackoverflow.com/questions/11030253/decision-values-in-libsvm

    predictor = np.inner(W, X_i) + threshold

    if (predictor <= 0):
        return -1
    else:
        return 1


def find_closest_point(point, X):
    # Returns index of closest point
    from scipy.spatial import distance
    distances = distance.cdist([point], X)
    sorted = np.argsort(distances)
    return sorted[0][1]


def multiclass_classifier(X, train_set, tie_mode=0):
    # Uses OVR approach for simplicity
    # If there are no classifier for an object, default to 1st class in Y
    # If there are ties, break by locality => assign class from closest point
    # tie mode = 0: assign directly class of closest point to classifying point
    # tie mode = 1: assign from the tie-ing classes that is the same as the closest point
    # If closest point is not classified yet -> wild guess from tie-ing classes

    # X = set of points to be classified
    # train_set = Already classified dataset (output from multiclass_train())

    X = np.array(X)
    # Y = np.array(Y)
    classifiers = train_set['classifiers']
    classes = train_set['classes']
    n_classes = len(classes)
    n_samples = len(X)
    ret = ['NA'] * n_samples

    for i in range(0, n_samples):
        votes = np.zeros(n_classes)
        for class_i in range(0, n_classes):
            votes[class_i] = binary_classifier(classifiers[class_i]['alpha'], classifiers[class_i]['threshold'],
                                               classifiers[class_i]['W'], X[i], X, classifiers[class_i]['Y_binary'])

        classified = np.where(votes == 1)[0]  # Assigned class is taken from 1st element in 'classified' array

        if len(classified) > 1:
            print("WARNING: " + str(len(classified)) + " ties in sample " + str(i))
            closest = find_closest_point(X[i], X)
            if ret[closest] != 'NA':  # Already classified
                if tie_mode == 0:
                    classified[0] = np.where(classes == ret[closest])[0][0]
                elif tie_mode == 1:
                    for j in classified:
                        if (ret[closest] == classes[j]):
                            classified[0] = j
                            break
            else:
                print("ERROR: Tie cannot be resolved!")
                classified[0] = 0

        if len(classified) == 0:
            print("WARNING: sample " + str(i) + " is rejected by all classes")
            closest_bound = math.inf
            classified = [0]
            for class_i in range(0, n_classes):
                dist2bound = abs(np.inner(classifiers[class_i]['W'], X[i]) + classifiers[class_i]['threshold'])
                if closest_bound >= dist2bound:
                    closest_bound = dist2bound
                    classified[0] = class_i

        ret[i] = classes[classified[0]]
    return ret


# X1 = np.array([[1, 3], [2, 2], [4, 3], [6, 1], [5, 4]]) - np.array([2, 2])
# X2 = np.array([[13, 6], [15, 8], [16, 7], [12, 9], [17, 10]]) + np.array([2, 2])
# X3 = np.array([[2, 7], [6, 10], [3, 8], [1, 9], [7, 8]]) + np.array([-2, 2])
# X4 = np.array([[12, 3], [13, 2], [16, 4], [15, 1], [11, 3]]) + np.array([2, -2])
# dataset = np.concatenate((X1, X2, X3, X4), axis=0)
# Y = np.array(
#     ['LL', 'LL', 'LL', 'LL', 'LL', 'UR', 'UR', 'UR', 'UR', 'UR', 'UL', 'UL', 'UL', 'UL', 'UL', 'LR', 'LR', 'LR', 'LR',
#      'LR'])
# X = np.array([[5, 2], [-1, 4], [1, 8], [4, 11], [15, -2], [16, 0], [18, 11], [12, 10]])
#
# train_set = multiclass_train(dataset, Y)
# Y_classified = np.array(multiclass_classifier(X, train_set))
#
# print("Labeled dataset: " + str(Y))
# print("Classified dataset: " + str(Y_classified))
