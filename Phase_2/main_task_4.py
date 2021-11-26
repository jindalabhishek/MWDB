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


def normal_dist(x):
    normal_x = (np.pi*np.std(x)) * np.exp(-0.5*((x-np.mean(x))/np.std(x))**2)
    return normal_x

def similar_matrix(data, method='pearson'):
    data = np.array(data)
    data[np.isnan(data)] = 0

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
            a=normal_dist(data[i])
            for j in range(0, data.shape[0]):
                b=normal_dist(data[j])
                d = np.sum(a * b) / np.sqrt(np.sum(np.square(a)) * np.sum(np.square(b)))
                matrix[i][j] = d
        return np.array(matrix)

def main():
    input_path = input("Enter subject weight features path: ")
    # input_path = "1_color_moment_feature_descriptor_PCA_1.json"

    subject_weights = {i[0]: i[1] for i in LatentSemanticFile.deserialize(input_path).task_output}
    subjects = [str(i) for i in range(1, 41)]

    subject_weights_array = numpy.array([subject_weights[key] for key in subjects])
    ##### Acquire parameters from user #####

    print("Calculating Similarity Matrix")

    similarity = similar_matrix(subject_weights_array)
    k = int(input("Enter k value: "))

    dimension_reduction_technique = input("Enter dimensionality reduction technique (1. PCA 2.SVD 3.LDA 4.k-means): ")
    #     dimension_reduction_technique = '3'
    technique_number_vs_reduction_object = {'1': PCA(), '2': SVD(), '3': LDA(), '4': KMeans(1000)}
    dimension_reduction_object = technique_number_vs_reduction_object[dimension_reduction_technique]

    latent_subject_features_dataset = dimension_reduction_object.compute(similarity, k, subjects)

    save_task_data('task_4', dimension_reduction_object, task_output=latent_subject_features_dataset.tolist())
    # print('type_weight_matrix dimension', len(type_weight_matrix), len(type_weight_matrix[0]))
    print('Entire Subject latent weight matrix: \n', latent_subject_features_dataset)

    k_subjects = np.transpose(latent_subject_features_dataset)
    for k in range(0, len(k_subjects)):
        print(k, 'th Semantic in Subject Weight Pairs')
        subject_weight_pairs = k_subjects[k]
        indexes = np.argsort(-subject_weight_pairs)
        for index in indexes:
            print(subjects[index], ':', subject_weight_pairs[index])


main()
