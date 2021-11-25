# Sources: 
# http://pzs.dstu.dp.ua/DataMining/svm/bibl/SuportVectorMachinesAndRelevanceFeedback.pdf

import numpy as np
import random
from sklearn import preprocessing
from matplotlib import pyplot as plt
from SVM import binary_train, binary_classifier
from VA_Files import VA_Approx, VA_SSA

# TODO: when there are no more objects to be labeled, at final iteration, function should find the best result to return to user (kNN that were labeled relevant)

def SVM_RF(X, Y, query, k=10):
  # Initial query: uses VA technique to get initial set for first classification
  # Suceeding feedbacks: 
  # Reconstruct support vectors when not all returned samples are labeled relevant
  # Return the samples that are classified as 1 and furthest from the support vectors
  # Stop when all returned samples are labeled relevant or user inputs "stop"
  # X = dataset of m features
  # Y = filename (labels) of corresponding objects in X -> should be parallel to X
  # Query = a data object of m features
  # k = number of nearest neighbors to query point

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
 
  # Get first feedback and train with SVM binary
  print("Initial query: " + str([Y[int(i)] for i in train_set_id]))
  user_ip = input("Enter feedback for each sample (-1 or 1), separated by ',': ")
  feedbacks = user_ip.split(',')
  labels = [int(i) for i in feedbacks]

  iter = 2
  while(len(np.unique(labels)) != 2):	# All samples are labeled the same
    train_set_id = VA_SSA(X, a, p, query, k*iter).tolist()
    train_set = [X[int(i)] for i in train_set_id]
    new_train_set_id = [train_set_id[i] for i in range(k*(iter-1), k*iter)]
    iter += 1
    print("Next initial query: " + str([Y[int(i)] for i in new_train_set_id]))
    user_ip = input("Enter feedback for each sample (-1 or 1), separated by ',': ")
    feedbacks = user_ip.split(',')
    labels.extend([int(i) for i in feedbacks])

  for i in range(0, n_samples):
    if i not in train_set_id: 
      test_set.append(X[i])  # NOT NEEDED - revise
      test_set_id.append(i)
  a, b, W = binary_train(np.array(train_set), np.array(labels))

  # Feedback iterations
  while(user_ip != 'stop'):
    # Calculate and find k objects in test_set that are furthest to the support vector
    #print("Test set id = " + str(test_set_id))
    test_set_dst = []
    for i in test_set_id:
      test_set_dst.append((np.inner(W, X[i]) + b))
    sorted_test_id = np.argsort(np.array(test_set_dst)*-1) # id in test_set_id to be sorted
    #print("test_set_dst = " + str(test_set_dst))
    #print("sorted_test_id = " + str(sorted_test_id))

    # Get train_set and feedback from user
    new_train_set = [X[test_set_id[sorted_test_id[i]]] for i in range(0, k)]
    new_train_set_id = [test_set_id[sorted_test_id[i]] for i in range(0, k)]
    print("Learned train set: " + str([Y[int(i)] for i in new_train_set_id]))
    user_ip = input("Enter feedback for each sample (-1 or 1), separated by ',', or enter 'all' for all relevant in the train set, or 'none' for all irrelevant in the train set, or 'stop' to stop training: ")
    if(user_ip == 'stop' or user_ip == 'all'): return
    if(user_ip == 'none'): labels.extend([-1]*k)
    else:
      feedbacks = user_ip.split(',')
      labels.extend([int(i) for i in feedbacks])
    print("Labels = " + str(labels))

    # Append new train_set to current train_set and do binary train
    train_set.extend(new_train_set)
    for i in new_train_set_id:
      test_set_id.remove(i) 	# And remove elements from the test_set
    a, b, W = binary_train(np.array(train_set), np.array(labels))

  return new_train_set_id


X1 = np.array([[1,3],[2,2],[4,3],[6,1],[5,4]])
X2 = np.array([[13,6],[15,8],[16,7],[12,9],[17,10]])
X3 = np.array([[2,7],[6,10],[3,8],[1,9],[7,8]])
X4 = np.array([[12,3],[13,2],[16,4],[15,1],[11,3]])
X = np.concatenate((X1, X2, X3, X4), axis = 0)
Y = ['LL','LL','LL','LL','LL','UR','UR','UR','UR','UR','UL','UL','UL','UL','UL','LR','LR','LR','LR','LR']
SVM_RF(X, Y, [10, 5], 2)
