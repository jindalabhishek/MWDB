# Sources: 
# http://pzs.dstu.dp.ua/DataMining/svm/bibl/SuportVectorMachinesAndRelevanceFeedback.pdf

import numpy as np
import random
from sklearn import preprocessing
from matplotlib import pyplot as plt
from SVM import binary_train, binary_classifier
from VA_Files import VA_Approx, VA_SSA

# TODO: account for the case when returned samples are all relevant in initial query and suceeding feedbacks
# TODO: returned samples should be shown in corresponding names to user to label
# TODO: function currently returns a new set of results that are not labeled before, it might be desired to consider previously labeled results as well
def SVM_RF(X, query, k=10):
  # Assumption: by nearest neighbors, find matches that are of the same class (relevant) as query point
  # Initial query: uses VA technique to get initial set for first classification
  # Suceeding feedbacks: 
  # Reconstruct support vectors when not all returned samples are labeled relevant
  # Return the samples that are classified as 1 and furthest from the support vectors
  # Stop when all returned samples are labeled relevant or user inputs "stop"

  n_samples = len(X)
  n_features = len(X[0])
  train_set = []   	# Set of samples already used for classification
  train_set_id = [] 	# Indices of samples that are in train_set
  test_set = []		# Set of samples to be classified and returned to user (NOT USED)
  test_set_id = []	# Indices of samples that are in test_set

  # Initial query
  a,p = VA_Approx(X, 4)
  train_set_id = VA_SSA(X, a, p, query, k).tolist()
  train_set = [X[int(i)] for i in train_set_id]
  for i in range(0, n_samples):
    if i not in train_set_id: 
      test_set.append(X[i])
      test_set_id.append(i)
  
  print("train_set = " + str(train_set) + "\ntrain_set_id = " + str(train_set_id))
  print("test_set = " + str(test_set) + "\ntest_set_id = " + str(test_set_id))

  # Get first feedback and train with SVM binary
  print("Initial query: " + str(train_set_id))
  user_ip = input("Enter feedback for corresponding sample above, separated by ',': ")
  feedbacks = user_ip.split(',')
  labels = [int(i) for i in feedbacks]
  a, b, W = binary_train(np.array(train_set), np.array(labels))

  # Feedback iterations
  while(user_ip != 'stop'):
    # Calculate and find k objects in test_set that are furthest to the support vector
    print("Test set id = " + str(test_set_id))
    test_set_dst = []
    for i in test_set_id:
      test_set_dst.append((np.inner(W, X[i]) + b))
    sorted_test_id = np.argsort(np.array(test_set_dst)*-1) # id in test_set_id to be sorted
    print("test_set_dst = " + str(test_set_dst))
    print("sorted_test_id = " + str(sorted_test_id))

    # Get train_set and feedback from user
    new_train_set = [X[test_set_id[sorted_test_id[i]]] for i in range(0, k)]
    new_train_set_id = [test_set_id[sorted_test_id[i]] for i in range(0, k)]
    print("Learned train set: " + str(new_train_set))
    user_ip = input("Enter feedback for corresponding sample above, separated by ',', or enter 'all' for all relevant in the train set, or 'stop' to stop training: ")
    if(user_ip == 'stop' or user_ip == 'all'): return
    feedbacks = user_ip.split(',')
    labels.extend([int(i) for i in feedbacks])
    print("Labels = " + str(labels))

    # Append new train_set to current train_set and do binary train
    train_set.extend(new_train_set)
    for i in new_train_set_id:
      test_set_id.remove(i) 	# And remove elements from the test_set
    a, b, W = binary_train(np.array(train_set), np.array(labels))

  return new_train_set_id
