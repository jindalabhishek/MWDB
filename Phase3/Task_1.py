import numpy as np
from file import *
from SVM import *
import PPR
from Decision_Tree import *
from dimensionality_reduction import *
import skimage.feature as skf
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
import Utils

# train_path = input("Enter the image folder path for training: ")
train_path = "/Users/swamirishi/Documents/asu/Fall_2021/MWDB/MWDB/images/500"
# feature_model =  input("Enter feature model technique ('CM', 'ELBP', 'HOG') : ")
feature_model = "CM"
# dimensions = int(input("Total reduced Dimensions: "))
dimensions = 20
dimension_reduction, trainFileNames = getTrainData(train_path, feature_model, dimensions, Utils.getType)
X_train, labels_train , trainFileNames = getTestData(train_path,feature_model,dimension_reduction,Utils.getType)

# classifier = input("Enter classifier model technique ('SVM', 'DT', 'PPR') : ")
classifier = "PPR"
# test_path = input("Enter the image folder path for testing: ")
test_path = "/Users/swamirishi/Documents/asu/Fall_2021/MWDB/MWDB/images/Train"

Y_train = labels_train  # types labels

X_test, labels_test, testFileNames = getTestData(test_path, feature_model, dimension_reduction,Utils.getType)
Y_test = labels_test # types labels

type2num = {'cc': 1, 'con': 2, 'emboss': 3, 'jitter': 4, 'neg': 5, 'noise01': 6, 'noise02': 7, 'original': 8,
            'poster': 9, 'rot': 10, 'smooth': 11, 'stipple': 12}
num2type = {1: 'cc', 2: 'con', 3: 'emboss', 4: 'jitter', 5: 'neg', 6: 'noise01', 7: 'noise02', 8: 'original',
            9: 'poster', 10: 'rot', 11: 'smooth', 12: 'stipple'}

if classifier == 'SVM':
    train_set = multiclass_train(np.array(X_train), np.array(Y_train))
    Y_hat = multiclass_classifier(X_test, train_set)

elif classifier == 'DT':
    data = X_train.copy()

    labels = list(map(lambda x: type2num[x], Y_train))

    data = np.concatenate((np.array(data), np.array([labels]).T), axis=1)
    tree = build_tree(data, 1000)
    la = []
    for inp in X_test:
        l = print_leaf(classify(inp, tree))
        k = l.popitem()
        la.append(int(k[0]))

    Y_hat = list(map(lambda x: num2type[x], la))

elif classifier == 'PPR':
    Y_hat = PPR.getTestingLabels(X_train, Y_train, X_test,Y_test, trainFileNames, testFileNames,13)

else:
    print('wrong classifier')
    exit(0)

calculate_and_print_results(Y_test, Y_hat, num2type)

