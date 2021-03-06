from file import *
from SVM import *
from Decision_Tree import *
import Utils
import PPR

train_path = input("Enter the image folder path for training: ")
feature_model = input("Enter feature model technique ('CM', 'ELBP', 'HOG') : ")
dimensions = int(input("Total reduced Dimensions: "))
print('Wait! Calculating latent semantics!')
dimension_reduction, trainFileNames = getTrainData(train_path, feature_model, dimensions, Utils.getSample)
X_train, labels_train, trainFileNames = getTestData(train_path,feature_model, dimension_reduction, Utils.getSample)
print('Successfully Finished computing latent semantics')
test_path = input("Enter the image folder path for testing: ")
classifier = input("Enter classifier model technique ('SVM', 'DT', 'PPR') : ")

Y_train = labels_train
X_test, labels_test, testFileNames = getTestData(test_path, feature_model, dimension_reduction, Utils.getSample)
Y_test = labels_test

type2num, num2type = {}, {}
for i in range(1, 11):
    type2num[str(i)] = i
    num2type[i] = str(i)

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
    Y_hat = PPR.getTestingLabels(X_train, Y_train, X_test, Y_test, trainFileNames, testFileNames, 15, Utils.getEuclideanDistance)

else:
    print('wrong classifier')
    exit(0)

calculate_and_print_results(Y_test, Y_hat, num2type)

