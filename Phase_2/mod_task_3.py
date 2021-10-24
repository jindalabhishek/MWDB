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
    latent_semantic_file_path = input("Enter latent semantic path: ")
    #     input_path = "2_color_moment_feature_descriptor_PCA_1.json"
    type_weights = {i[0]:i[1] for i in get_task_output_data(latent_semantic_file_path)}
    types = [tt for tt in type_weights]

    type_weights = numpy.array([type_weights[tt] for tt in type_weights])
    ##### Acquire parameters from user #####
    similarity = similar_matrix(type_weights)
    print("Calculating Similarity Matrix")

    k = int(input("Enter k value: "))
    dimension_reduction_technique = input("Enter dimensionality reduction technique (1. PCA 2.SVD 3.LDA 4.k-means): ")

    technique_number_vs_reduction_object = {'1': PCA(), '2': SVD(), '3': LDA(), '4': KMeans(1000)}
    dimension_reduction_object = technique_number_vs_reduction_object[dimension_reduction_technique]

    #     dimension_reduction_technique = '3'
    latent_type_features_dataset = dimension_reduction_object.compute(similarity, k, types)
    save_task_data('task_3', dimension_reduction_object, task_output=latent_type_features_dataset.tolist())
    # print('type_weight_matrix dimension', len(type_weight_matrix), len(type_weight_matrix[0]))
    print('Entire Type-Type similarity weight matrix: \n', latent_type_features_dataset)


main()
