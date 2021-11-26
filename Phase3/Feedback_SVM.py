# Sources: 
# http://pzs.dstu.dp.ua/DataMining/svm/bibl/SuportVectorMachinesAndRelevanceFeedback.pdf

import numpy as np
import random
from sklearn import preprocessing
from matplotlib import pyplot as plt
from SVM import binary_train, binary_classifier
from VA_Files import VA_Approx, VA_SSA


# TODO: when there are no more objects to be labeled, at final iterations,
#  function should find the best result to return to user (kNN that were labeled relevant)

def SVM_RF(image_vector_matrix, image_labels, query, number_of_bits, k=10):
    # Initial query: uses VA technique to get initial set for first classification
    # Succeeding feedbacks:
    # Reconstruct support vectors when not all returned samples are labeled relevant
    # Return the samples that are classified as 1 and furthest from the support vectors
    # Stop when all returned samples are labeled relevant or user inputs "stop"
    # image_vector_matrix = dataset of m features
    # image_labels = filename (labels) of corresponding objects in image_vector_matrix -> should be parallel to image_vector_matrix
    # Query = a data object of m features
    # k = number of nearest neighbors to query point

    n_samples = len(image_vector_matrix)
    test_set = []  # Set of samples to be classified and returned to user (NOT USED)
    test_set_id = []  # Indices of samples that are in test_set

    # Initial query
    approximations, partition_boundaries = VA_Approx(image_vector_matrix, number_of_bits)

    # Indices of samples that are in train_set
    train_set_id = VA_SSA(image_vector_matrix, approximations, partition_boundaries, query, k).tolist()
    # Set of samples already used for classification
    train_set = [image_vector_matrix[int(i)] for i in train_set_id]

    # Get first feedback and train with SVM binary
    print("Initial query: " + str([image_labels[int(i)] for i in train_set_id]))
    user_ip = input("Enter feedback for each sample irrelevant(-1) or relevant(1), separated by ',': ")
    feedbacks = user_ip.split(',')
    labels = [int(i) for i in feedbacks]

    # Get New Feedbacks in case user marked all images as relevant or irrelevant
    iteration = 2
    while len(np.unique(labels)) != 2:  # All samples are labeled the same
        train_set_id = VA_SSA(image_vector_matrix, approximations, partition_boundaries, query, k * iteration).tolist()
        train_set = [image_vector_matrix[int(i)] for i in train_set_id]
        new_train_set_id = [train_set_id[i] for i in range(k * (iteration - 1), k * iteration)]
        iteration += 1
        print("Next initial query: " + str([image_labels[int(i)] for i in new_train_set_id]))
        user_ip = input("Enter feedback for each sample (-1 or 1), separated by ',': ")
        feedbacks = user_ip.split(',')
        labels.extend([int(i) for i in feedbacks])

    for i in range(0, n_samples):
        # if i not in train_set_id:
        test_set.append(image_vector_matrix[i])  # NOT NEEDED - revise
        test_set_id.append(i)
    a, b, W = binary_train(np.array(train_set), np.array(labels))

    # Feedback iterations
    while user_ip != 'stop':
        # Calculate and find k objects in test_set that are furthest to the support vector
        # print("Test set id = " + str(test_set_id))
        test_set_dst = []
        for i in test_set_id:
            test_set_dst.append((np.inner(W, image_vector_matrix[i]) + b))
        print(test_set_dst)
        sorted_test_id = np.argsort(np.array(test_set_dst) * -1)  # id in test_set_id to be sorted
        # print("test_set_dst = " + str(test_set_dst))
        # print("sorted_test_id = " + str(sorted_test_id))

        # Get train_set and feedback from user
        new_train_set = [image_vector_matrix[test_set_id[sorted_test_id[i]]] for i in range(0, k)]
        new_train_set_id = [test_set_id[sorted_test_id[i]] for i in range(0, k)]
        print("Learned train set: " + str([image_labels[int(i)] for i in new_train_set_id]))
        print("Enter feedback for each sample (-1 or 1), separated by ',', or enter 'all'\n"+
              "for all relevant in the train set, or 'none' for all irrelevant in the train set, \n"
              "or 'stop' to stop training: ")
        user_ip = input()
        if user_ip == 'stop' or user_ip == 'all':
            return
        if user_ip == 'none':
            labels.extend([-1] * k)
        else:
            feedbacks = user_ip.split(',')
            labels.extend([int(i) for i in feedbacks])
        print("Labels = " + str(labels))

        # Append new train_set to current train_set and do binary train
        train_set.extend(new_train_set)
        # for i in new_train_set_id:
        #     test_set_id.remove(i)  # And remove elements from the test_set
        a, b, W = binary_train(np.array(train_set), np.array(labels))

    return new_train_set_id

# image_vector_matrix1 = np.array([[1, 3], [2, 2], [4, 3], [6, 1], [5, 4]])
# image_vector_matrix2 = np.array([[13, 6], [15, 8], [16, 7], [12, 9], [17, 10]])
# image_vector_matrix3 = np.array([[2, 7], [6, 10], [3, 8], [1, 9], [7, 8]])
# image_vector_matrix4 = np.array([[12, 3], [13, 2], [16, 4], [15, 1], [11, 3]])
# image_vector_matrix = np.concatenate((image_vector_matrix1, image_vector_matrix2, image_vector_matrix3, image_vector_matrix4), axis=0)
# image_labels = ['LL', 'LL', 'LL', 'LL', 'LL', 'UR', 'UR', 'UR', 'UR', 'UR', 'UL', 'UL', 'UL', 'UL', 'UL', 'LR', 'LR', 'LR', 'LR',
#      'LR']
# SVM_RF(image_vector_matrix, image_labels, [10, 5], 2)
