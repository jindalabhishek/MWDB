from matrix_util import *


def transform_1xm_to_1xk(matrix_1xm, all_latent_semantics, feature_model):
    matrix_1xk = []
    if feature_model == 'pca':
        matrix_1xk = multiply_matrices(matrix_1xm, all_latent_semantics['matrix_mxk'])
        matrix_1xk = multiply_matrices(matrix_1xk, inverse_matrix(all_latent_semantics['matrix_kxk']))
    elif feature_model == 'svd':
        matrix_1xk = multiply_matrices(transpose_matrix(all_latent_semantics['matrix_kxm']))
        matrix_1xk = multiply_matrices(matrix_1xk, inverse_matrix(all_latent_semantics['matrix_kxk']))
    elif feature_model == 'lda':
        pass
    elif feature_model == 'k-means':
        matrix_kxm = all_latent_semantics['centroids_kxm']
        # Euclidean distance between query and all centroids
        for each in matrix_kxm:
            matrix_1xk.append(np.linalg.norm(matrix_1xm - each))

    return matrix_1xk


def get_reduced_dimension_nxk_using_latent_semantics(data_matrix_nxm, all_latent_semantics, feature_model):
    reduced_matrix_nxk = []
    for each in data_matrix_nxm:
        reduced_matrix_nxk.append(transform_1xm_to_1xk(each, all_latent_semantics, feature_model))

    return reduced_matrix_nxk
