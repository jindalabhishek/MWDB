# Sources: 
# https://www.researchgate.net/publication/2436132_Relevance_Feedback_Decision_Trees_in_Content-Based_Image_Retrieval

import numpy as np
import random
from sklearn import preprocessing
from matplotlib import pyplot as plt

from LSH import LSHash
from Decision_Tree import *


def DT_RF(X, data_labels, query, k_n=10):
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
    number_of_bits = int(input('Enter the number of hash functions in a family of hash functions: '))
    number_of_families = int(input('Enter the number of hash families: '))
    LSH = LSHash(n_features, k_bit_hash=number_of_bits, num_hashtables=number_of_families)
    # X = np.random.randn(130, 3)
    # inps = input array
    for inp,label in zip(X,data_labels):
        LSH.index(inp,label)


    X_list = X.tolist()

    train_set,_ = LSH.query(query, k_n)
    train_set = train_set.tolist()
    test_set = X.tolist()
    if not (isinstance(data_labels, list)):
        data_labels = data_labels.tolist()

    labels_of_similar_images = []
    for i in range(0, len(train_set)):
        labels_of_similar_images.append(data_labels[i])
    print(labels_of_similar_images)

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
    # for x in train_set:1,0,1,0,1,0,1,0,1,0
    #   test_set.remove(x)
    #print(np.asarray(train_set)[0].shape)

    tree = build_tree(np.concatenate((np.asarray(train_set), np.asarray([labels]).T), axis=1), 1000)

    # Feedback iterations

    while user_ip != 'stop':
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

        new_train_set,_ = LSH1.query(query, k_n)
        # print(new_train_set,'==============================')

        new_train_set = new_train_set.tolist()
        labels_of_similar_images = []
        for x in new_train_set:
            labels_of_similar_images.append(data_labels[X_list.index(x)])
        print(labels_of_similar_images)

        user_ip = input(
            "Enter feedback for corresponding sample above, separated by ',', or enter 'all' for all relevant "
            "in the train set, or 'stop' to stop training: ")
        if user_ip == 'stop' or user_ip == 'all':
            return np.array(new_train_set)
        feedbacks = user_ip.split(',')
        labels.extend([int(i) for i in feedbacks])
        print("Labels = " + str(labels))

        # Append new train_set to current train_set and do binary train
        train_set.extend(new_train_set)
        # for i in new_train_set:
        #   test_set.remove(i) 	# And remove elements from the test_set
        tree = build_tree(np.concatenate((np.array(train_set,dtype=object), np.array([labels],dtype=object).T), axis=1), 1000)

    return np.array(new_train_set)
