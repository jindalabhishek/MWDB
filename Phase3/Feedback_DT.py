# Sources: 
# https://www.researchgate.net/publication/2436132_Relevance_Feedback_Decision_Trees_in_Content-Based_Image_Retrieval

import numpy as np
import random
from sklearn import preprocessing
from matplotlib import pyplot as plt

from LSH import LSHash
from Decision_Tree import *


def DT_RF(X, query, k=10):
    # Assumption: by nearest neighbors, find matches that are of the same class (relevant) as query point
    # Initial query: uses LSH technique to get initial set for first classification
    # Suceeding feedbacks:
    # Reconstruct support vectors when not all returned samples are labeled relevant
    # Return the samples that are classified as 1 and furthest from the support vectors
    # Stop when all returned samples are labeled relevant or user inputs "stop"

    n_samples = len(X)
    n_features = len(X[0])
    train_set = []  # Set of samples already used for classification
    train_set_id = []  # Indices of samples that are in train_set
    test_set = []  # Set of samples to be classified and returned to user (NOT USED)
    test_set_id = []  # Indices of samples that are in test_set
    # Initial query
    LSH = LSHash(n_features, k_bit_hash=4, num_hashtables=2)
    # X = np.random.randn(130, 3)
    # inps = input array
    for inp in X:
        LSH.index(inp)

    train_set = LSH.query(query, k)
    test_set = X
    # for x in train_set:
    #   test_set.remove(x)

    if isinstance(query, np.ndarray):
        query = query.tolist()
    train_set.append(query)

    # Get first feedback and train with DT binary

    user_ip = input("Enter feedback for corresponding sample above, separated by ',': ")
    feedbacks = user_ip.split(',')
    labels = [int(i) for i in feedbacks]
    labels.append(1)

    # iter = 2
    # while(len(np.unique(labels)) != 2):	# All samples are labeled the same
    #   train_set = LSH.query(query_point, k*iter)
    #   train_set = train_set.tolist()
    #   new_train_set_id = [train_set_id[i] for i in range(k*(iter-1), k*iter)]
    #   iter += 1
    #
    #   user_ip = input("Enter feedback for each sample (-1 or 1), separated by ',': ")
    #   feedbacks = user_ip.split(',')
    #   labels.extend([int(i) for i in feedbacks])

    # test_set = X.tolist()
    # for x in train_set:
    #   test_set.remove(x)
    # train_set.append(query_point.tolist())

    tree = build_tree(np.concatenate((np.array(train_set), np.array([labels]).T), axis=1), 1000)

    # Feedback iterations

    while (user_ip != 'stop'):
        # Calculate and find k objects in test_set through DT
        test_set_dst = []
        for inp in test_set:
            if int(list(print_leaf(classify(inp, tree)).keys())[0]) == 1:
                test_set_dst.append(inp)

        LSH1 = LSHash(n_features, k_bit_hash=4, num_hashtables=4)
        X = np.array(test_set_dst)
        # inps = input array
        for inp in X:
            LSH1.index(inp)

        new_train_set = LSH1.query(query, k)
        print(new_train_set, '==============================')
        new_train_set = new_train_set.tolist()
        user_ip = input(
            "Enter feedback for corresponding sample above, separated by ',', or enter 'all' for all relevant in the train set, or 'stop' to stop training: ")
        if (user_ip == 'stop' or user_ip == 'all'): return np.array(new_train_set)
        feedbacks = user_ip.split(',')
        labels.extend([int(i) for i in feedbacks])
        print("Labels = " + str(labels))

        # Append new train_set to current train_set and do binary train
        train_set.extend(new_train_set)
        # for i in new_train_set:
        #   test_set.remove(i) 	# And remove elements from the test_set
        tree = build_tree(np.concatenate((np.array(train_set), np.array([labels]).T), axis=1), 1000)

    return np.array(new_train_set)
