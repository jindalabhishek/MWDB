import numpy
from Phase_2.dimensionality_reduction.KMeans import KMeans
from Phase_2.dimensionality_reduction.LDA import LDA
from Phase_2.dimensionality_reduction.PCA import PCA
from Phase_2.dimensionality_reduction.SVD import SVD
from Util.Utils import get_output_file_path
from Util.dao_util import DAOUtil
from numpy.linalg import svd
import numpy as np
from Util.json_util import LatentSemanticFile


def similar_matrix(data, method='pearson'):
    data = np.array(data, dtype=np.float128)
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
    # input_path = input("Enter type weight features path: ")
    input_path = "1_color_moment_feature_descriptor_PCA_1.json"
    subject_weights = LatentSemanticFile.deserialize(input_path).task_output
    subjects = [ss for ss in subject_weights]

    subject_weights = numpy.array([subject_weights[ss] for ss in subject_weights])
    ##### Acquire parameters from user #####
    
    print("Calculating Similarity Matrix")

    similarity = similar_matrix(subject_weights)

    k = int(input("Enter k value: "))


    reductionTechniqueDict = {'1': PCA(), '2': SVD(), '3': LDA(), '4': KMeans(1000)}

    subjects = [subject for subject in {str(i) for i in range(1, 41)}]
    # dimension_reduction_technique = input("Enter dimensionality reduction technique: ")
    dimension_reduction_technique = '3'
    latent_subject_features_dataset = reductionTechniqueDict[dimension_reduction_technique].compute((similarity), k, subjects)

    LatentSemanticFile("dummy", reductionTechniqueDict[dimension_reduction_technique],
                       latent_subject_features_dataset.tolist()) \
        .serialize(
        get_output_file_path(4, "", "input_path", type(reductionTechniqueDict[dimension_reduction_technique]).__name__))
    # print('type_weight_matrix dimension', len(type_weight_matrix), len(type_weight_matrix[0]))
    print('Entire Type-Type similarity weight matrix: \n', latent_subject_features_dataset)

    return latent_subject_features_dataset, k
main()
