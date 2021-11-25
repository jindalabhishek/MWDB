# Task 1
import numpy as np
from readData import *
from SVM import *
from DecisionTree import *
from Dimensionality_Reduction import *
import skimage.feature as skf



feature_model = input("Enter feature model (color, lbp or hog): ")
dimensionality_technique= input("Enter dimensionality technique (pca, svd, lda or kmeans): ")
k = int(input("Enter value k: "))
classification_technique = input("Enter classification technique (svm, dt or ppr): ")

X, Y = readData()
n_samples = len(X)
for i in range(n_samples):
  Y[i] = Y[i].split('-')[1]
  X[i] = X[i].flatten()
  
X_ls = lda_decomposition(np.array(X), k)

if(classification_technique == 'svm'):
  train_set = multiclass_train(np.array(X_ls), np.array(Y))
  Yc = multiclass_classifier(X_ls, train_set)
elif(classification_technique == 'dt'):
  X_ll = X_ls.tolist()
  for i in range(len(X_ll)):
    X_ll[i].append(Y[i])
  tree = build_tree(X_ll, 6)
  Yc = []
  for i in range(len(X_ll)):
    Yc.append(list(classify(X_ll[i], tree).keys())[0])

c = 0
for i in range(len(Y)):
 if(Y[i] == Yc[i]): c+=1
