import numpy as np
from file import *
from SVM import *
from DecisionTree import *
from Dimensionality_Reduction import *
import skimage.feature as skf
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix

train_path = input("Enter the image folder path for training: ")
feature_model =  input("Enter feature model technique ('color', 'elbp', 'hog') : ")
dimensions = input("Total reduced Dimensions: ")

classifier = input("Enter classifier model technique ('SVM', 'DT', 'PPR') : ")

test_path = input("Enter the image folder path for testing: ")

X_train, labels_train = retrive_data(train_path, feature_model, dimensions)
Y_train = labels_train[0] # types labels

X_test, labels_test = retrive_data(train_path, feature_model, dimensions)
Y_test = labels_test[0] # types labels



if(classifier == 'SVM'):

  train_set = multiclass_train(np.array(X_train), np.array(Y_train))
  Y_hat = multiclass_classifier(X_test, train_set)

elif(classifier == 'DT'):


  data = X_train.copy()

  type2num={'cc':1, 'con':2, 'emboss':3, 'jitter':4, 'neg':5, 'noise01':6, 'noise02':7, 'original':8, 'poster':9, 'rot':10, 'smooth':11, 'stipple':12}
  num2type={1:'cc', 2:'con', 3:'emboss', 4:'jitter', 5:'neg', 6:'noise01', 7:'noise02', 8:'original', :9:'poster', 10:'rot', 11:'smooth', 12'stipple'}

  labels = list(map(lambda x: type2num[x], Y_train))

  data = np.concatenate((np.array(data),np.array([labels]).T), axis=1)
  tree = build_tree(data,1000)
  la=[]
  for inp in X_test:
	  l=print_leaf(classify(inp, tree))
	  k=l.popitem()
	  la.append(int(k[0]))
   
  Y_hat = list(map(lambda x: num2type[x], la))

elif (classifier == 'PPR'):
  # Dhruv code

else:
  print('wrong classifier')
  exit(0)



cm=multilabel_confusion_matrix(yhat, ok,labels=list(num2type.values()),)
fp={}
fn={}
total_fp, total_fn = 0, 0
for i in range(len(y.values())):
  fp[list(y.values())[i]] = cm[i][0][1]
  misses[list(y.values())[i]] = cm[i][1][0]
  total_fp+=cm[i][0][1]
  total_misses+=cm[i][1][0]

fp,misses,total_fp,total_misses
print('Total false positives = ', total_fp)
print(fp)
print('Total false positives = ', total_misses)
print(misses)
