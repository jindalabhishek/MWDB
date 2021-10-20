import numpy

from Util.dao_util import DAOUtil
from numpy.linalg import svd
from dimention_reduction_util import *

def get_image_vector_matrix(feature_descriptors, feature_model):
    image_vector_matrix = []
    image_types = []
    for feature_descriptor in feature_descriptors:
        image_vector_matrix.append(feature_descriptor[feature_model])
        image_types.append(feature_descriptor[DAOUtil.LABEL].split("-")[2])
    return image_vector_matrix,image_types


def main():
    """
        Executes Task 2
        Output Subject - latent semantics matrix, (subject-list of weight matrix)
    """
    """
        Connection to MongoDB using PyMongo
    """
    dao_util = DAOUtil()
    # feature_model = input('Welcome to Task 1 Demo. Enter the feature model:')
    feature_model = 'hog'
    feature_model += '_feature_descriptor'
    # type_id = input('Enter Type Id:')
    type_id = 'cc'
    feature_descriptors = dao_util.get_feature_descriptors_by_type_id(type_id)
    image_vector_matrix,image_types = get_image_vector_matrix(feature_descriptors, feature_model)
    # dimension_reduction_technique = input('Select Dimension reduction technique: (1. PCA 2.SVD 3.LDA 4.k-means): ')
    dimension_reduction_technique = '2'
    print('Image_vector_matrix dimension: ', len(image_vector_matrix),len(image_vector_matrix[0]))
    image_vector_matrix = numpy.array(image_vector_matrix)
    if dimension_reduction_technique == '1':
        subject_weight_matrix = get_reduced_matrix_using_pca(image_vector_matrix, 13)
    elif dimension_reduction_technique == '2':
        subject_weight_matrix = get_reduced_matrix_using_svd(image_vector_matrix,image_types, 13)

    print('Subject_weight_matrix dimension', len(subject_weight_matrix), len(subject_weight_matrix[0]))
    print('Entrire Subject weight matrix: \n', subject_weight_matrix)


main()
