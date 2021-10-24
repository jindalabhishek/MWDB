import numpy
from Phase_2.dimensionality_reduction.KMeans import KMeans
from Phase_2.dimensionality_reduction.LDA import LDA
from Phase_2.dimensionality_reduction.PCA import PCA
from Phase_2.dimensionality_reduction.SVD import SVD
from Util.Utils import *
from Util.dao_util import DAOUtil
from numpy.linalg import svd
import numpy as np
from Util.json_util import LatentSemanticFile


def similar_matrix(data, method='pearson'):
    data = np.array(data)
    matrix = np.ones((data.shape[0], data.shape[0]))
    if method == 'euclidean':
        for i in range(0, data.shape[0]):
            for j in range(0, data.shape[0]):
                d = (sum((data[i] - data[j]) ** 2)) ** 0.5
                d = 1 / (1 + d)
                matrix[i][j] = d
        return np.array(matrix)
    elif method == 'pearson':
        matrix = np.ones((data.shape[0], data.shape[0]))
        L = (data.T - data.mean(axis=1)).T
        for i in range(0, data.shape[0]):
            for j in range(0, data.shape[0]):
                d = np.sum(L[i] * L[j]) / np.sqrt(np.sum(np.square(L[i])) * np.sum(np.square(L[j])))
                matrix[i][j] = d
        return np.array(matrix)


def main():
    input_path = input("Enter type weight features path: ")
    # input_path = "1_color_moment_feature_descriptor_PCA_1.json"
    subject_weights = LatentSemanticFile.deserialize(input_path).task_output
    subjects = [ss for ss in subject_weights]

    subject_weights = numpy.array([subject_weights[ss] for ss in subject_weights])
    ##### Acquire parameters from user #####

    print("Calculating Similarity Matrix")

    similarity = similar_matrix(subject_weights)
    k = int(input("Enter k value: "))

    subjects = [subject for subject in {str(i) for i in range(1, 41)}]
    dimension_reduction_technique = input("Enter dimensionality reduction technique: ")
    #     dimension_reduction_technique = '3'
    technique_number_vs_reduction_object = {'1': PCA(), '2': SVD(), '3': LDA(), '4': KMeans(1000)}
    dimension_reduction_object = technique_number_vs_reduction_object[dimension_reduction_technique]

    latent_subject_features_dataset = dimension_reduction_object.compute(similarity, k, subjects)

    save_task_data('task_4', dimension_reduction_object, task_output=latent_subject_features_dataset)
    # print('type_weight_matrix dimension', len(type_weight_matrix), len(type_weight_matrix[0]))
    print('Entire Subject-Subject similarity weight matrix: \n', latent_subject_features_dataset)


main()
