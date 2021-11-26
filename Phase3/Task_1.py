import numpy as np
from file import *
from SVM import *
from Decision_Tree import *
import skimage.feature as skf
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
from dao_util import DAOUtil


def get_image_vector_matrix(feature_descriptors, feature_model):
    image_vector_matrix = []
    image_labels = []
    for feature_descriptor in feature_descriptors:
        image_vector_matrix.append(feature_descriptor[feature_model])
        image_labels.append(feature_descriptor['label'])
    return image_vector_matrix, image_labels


def main():
    dao_util = DAOUtil()
    feature_model = input('Welcome to Task 1 Demo. Enter the feature model (color_moment, elbp, hog):')
    dimensions = int(input("Total reduced Dimensions: "))

    training_feature_descriptors = dao_util.get_feature_descriptors_for_all_images()
    training_image_vector_matrix, training_image_labels = get_image_vector_matrix(training_feature_descriptors,
                                                                                  feature_model)

    classifier = input("Enter classifier model technique ('SVM', 'DT', 'PPR') : ")

    test_feature_descriptors = dao_util.get_feature_descriptors_for_all_images()
    test_image_vector_matrix, test_image_labels = get_image_vector_matrix(test_feature_descriptors, feature_model)

    if classifier == 'SVM':
        train_set = multiclass_train(np.array(training_image_vector_matrix), np.array(training_image_labels))
        Y_hat = multiclass_classifier(test_image_vector_matrix, train_set)

    elif classifier == 'DT':

        data = training_image_vector_matrix.copy()

        type2num = {'cc': 1, 'con': 2, 'emboss': 3, 'jitter': 4, 'neg': 5, 'noise01': 6, 'noise02': 7, 'original': 8,
                    'poster': 9, 'rot': 10, 'smooth': 11, 'stipple': 12}
        num2type = {1: 'cc', 2: 'con', 3: 'emboss', 4: 'jitter', 5: 'neg', 6: 'noise01', 7: 'noise02', 8: 'original',
                    9: 'poster', 10: 'rot', 11: 'smooth', 12: 'stipple'}

        labels = list(map(lambda x: type2num[x], training_image_labels))

        data = np.concatenate((np.array(data), np.array([labels]).T), axis=1)
        tree = build_tree(data, 1000)
        la = []
        for inp in test_image_vector_matrix:
            l = print_leaf(classify(inp, tree))
            k = l.popitem()
            la.append(int(k[0]))

        Y_hat = list(map(lambda x: num2type[x], la))

    elif classifier == 'PPR':
        k = 1

    else:
        print('wrong classifier')
        exit(0)

    cm = multilabel_confusion_matrix(test_image_labels, Y_hat, labels=list(num2type.values()))
    fp = {}
    misses = {}
    total_fp, total_fn, total_misses = 0, 0, 0
    for i in range(len(num2type.values())):
        fp[list(num2type.values())[i]] = cm[i][0][1]
        misses[list(num2type.values())[i]] = cm[i][1][0]
        total_fp += cm[i][0][1]
        total_misses += cm[i][1][0]

    print('Total false positives = ', total_fp)
    print(fp)
    print('Total misses = ', total_misses)
    print(misses)


main()

