from Util.dao_util import DAOUtil
from numpy.linalg import svd
from sklearn.decomposition import TruncatedSVD
from dimention_reduction_util import *


def get_image_vector_matrix(feature_descriptors, feature_model):
    image_vector_matrix = []
    for feature_descriptor in feature_descriptors:
        image_vector_matrix.append(feature_descriptor[feature_model])
    return image_vector_matrix


def main():
    """
        Executes Task 2
        Output Subject - latent semantics matrix, (subject-list of weight matrix)
    """
    """
        Connection to MongoDB using PyMongo
    """
    dao_util = DAOUtil()
    feature_model = input('Welcome to Task 2 Demo. Enter the feature model:')
    feature_model += '_feature_descriptor'
    subject_id = input('Enter Subject Id:')
    feature_descriptors = dao_util.get_feature_descriptors_by_subject_id(subject_id)
    image_vector_matrix = get_image_vector_matrix(feature_descriptors, feature_model)
    dimension_reduction_technique = input('Select Dimension reduction technique: (1. PCA 2.SVD 3.LDA 4.k-means): ')
    print('Image_vector_matrix dimension: ', len(image_vector_matrix),len(image_vector_matrix[0]))
    if dimension_reduction_technique == '1':
        subject_weight_matrix = get_reduced_matrix_using_pca(image_vector_matrix, 13)
    elif dimension_reduction_technique == '2':
        subject_weight_matrix = get_reduced_matrix_using_svd(image_vector_matrix, 13)

    print('Subject_weight_matrix dimension', len(subject_weight_matrix), len(subject_weight_matrix[0]))
    print('Entrire Subject weight matrix: \n', subject_weight_matrix)


main()
