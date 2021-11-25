import numpy as np
from file import *
from SVM import *
from DecisionTree import *
from Dimensionality_Reduction import *
import skimage.feature as skf
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix

train_path = input("Enter the image folder path for training: ")
feature_model =  input("Enter feature model technique ('CM', 'ELBP', 'HOG') : ")
dimensions = int(input("Total reduced Dimensions: "))

X_train, labels_train = retrive_data(train_path, feature_model, dimensions)

classifier = input("Enter classifier model technique ('SVM', 'DT', 'PPR') : ")

test_path = input("Enter the image folder path for testing: ")


Y_train = labels_train[1] # subject labels

X_test, labels_test = retrive_data(train_path, feature_model, dimensions)
Y_test = labels_test[1] # subject labels



if(classifier == 'SVM'):

  train_set = multiclass_train(np.array(X_train), np.array(Y_train))
  Y_hat = multiclass_classifier(X_test, train_set)

elif(classifier == 'DT'):


  data = X_train.copy()

  type2num,num2type={},{}
  for i in range(1,41):
    type2num[str(i)]=i
    num2type[i]=str(i)

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
  k=1

else:
  print('wrong classifier')
  exit(0)



cm=multilabel_confusion_matrix(Y_test, Y_hat,labels=list(num2type.values()))
fp={}
misses={}
tp={}
tn={}
total_fp, total_fn, total_tp, total_tn= 0, 0,0,0
for i in range(len(num2type.values())):
  fp[list(num2type.values())[i]] = cm[i][0][1]
  misses[list(num2type.values())[i]] = cm[i][1][0]
  tp[list(num2type.values())[i]] = cm[i][1][1]
  tn[list(num2type.values())[i]] = cm[i][0][0]
  total_fp+=cm[i][0][1]
  total_tp+=cm[i][1][1]
  total_tn+=cm[i][0][0]
  total_misses+=cm[i][1][0]

fp,misses,total_fp,total_misses
print('Total false positives = ', total_fp)
print(fp)
print('Total misses = ', total_misses)
print(misses)
print('Total correctly classified = ', total_tp)
print(tp)
