from Util.dao_util import DAOUtil
from numpy.linalg import svd
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
from Util.k_means_util import reduce_dimensions_k_means
import numpy as np

from sklearn.decomposition import LatentDirichletAllocation
from Util.k_means_util import reduce_dimensions_k_means
import numpy as np


def get_image_vector_matrix(feature_descriptors, feature_model):
    image_vector_matrix = []
    for feature_descriptor in feature_descriptors:
        image_vector_matrix.append(feature_descriptor[feature_model])
    return image_vector_matrix


def get_image_label_array(feature_descriptors):
    image_label_array = []
    for feature_descriptor in feature_descriptors:
        label = feature_descriptor['label']
        image_type = label.split('-')[1]
        image_label_array.append(image_type)
        # image_label_array.append(i)
    return image_label_array


def calculate_lda(image_vector_matrix, image_types, components):
    lda = LatentDirichletAllocation(n_components=components)
    image_vector_matrix_lda = lda.fit_transform(image_vector_matrix, image_types)
    return image_vector_matrix_lda


def normalize_data_for_lda(image_vector_matrix):
    normalized_data = (image_vector_matrix - np.min(image_vector_matrix)) \
                      / (np.max(image_vector_matrix) - np.min(image_vector_matrix))
    return normalized_data


def main():
    """
        Executes Task 2
        Output Subject - latent semantics matrix, (subject-list of weight matrix)
    """
    """
        Connection to MongoDB using PyMongo
    """
    dao_util = DAOUtil()
    feature_model = input('Welcome to Task 2 Demo. Enter the feature model (color_moment, elbp, hog):')
    feature_model += '_feature_descriptor'
    subject_id = input('Enter Subject Id (Y):')
    dimension_reduction_technique = input('Enter dimension reduction technique (pca, svd, lda, kmeans):')
    k = int(input('Enter value of k:'))
    feature_descriptors = dao_util.get_feature_descriptors_by_subject_id(subject_id)
    image_vector_matrix = get_image_vector_matrix(feature_descriptors, feature_model)
    image_types = get_image_label_array(feature_descriptors)
    max_n_components = len(set(image_types))
    print("Max N Components:", max_n_components)
    k = min(k, max_n_components)
    image_vector_matrix_k_dimensions = None
    if dimension_reduction_technique == 'pca':
        pass
    elif dimension_reduction_technique == 'svd':
        pass
    elif dimension_reduction_technique == 'lda':
        normalized_data = normalize_data_for_lda(np.array(image_vector_matrix))
        image_vector_matrix_k_dimensions = calculate_lda(normalized_data, image_types, k)
    elif dimension_reduction_technique == 'kmeans':
        image_vector_matrix_k_dimensions = reduce_dimensions_k_means(image_vector_matrix,
                                                                     n_components=k, n_iterations=1000)
    print(image_vector_matrix_k_dimensions.shape)
    # print(len(image_vector_matrix_k_dimensions))
    # print(len(image_vector_matrix_k_dimensions[0]))


main()
