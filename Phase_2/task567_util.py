from matrix_util import *


def transform_1xm_to_1xk(matrix_1xm, all_latent_semantics):
    matrix_1xk = []
    method_dimension_reduction = all_latent_semantics['reduction_technique']
    latent_semantics = all_latent_semantics['latent_features']
    if method_dimension_reduction == 'PCA':
        matrix_1xk = multiply_matrices(matrix_1xm, latent_semantics['matrix_mxk'])
        matrix_1xk = multiply_matrices(matrix_1xk, inverse_matrix(latent_semantics['matrix_kxk']))
    elif method_dimension_reduction == 'SVD':
        matrix_1xk = multiply_matrices(matrix_1xm, latent_semantics['matrix_mxk'])
        matrix_1xk = multiply_matrices(matrix_1xk, inverse_matrix(np.diagflat(latent_semantics['matrix_kxk'])))
    elif method_dimension_reduction == 'LDA':
        pass
    elif method_dimension_reduction == 'k-means':
        matrix_kxm = latent_semantics['centroids_kxm']
        # Euclidean distance between query and all centroids
        for each in matrix_kxm:
            matrix_1xk.append(np.linalg.norm(matrix_1xm - each))

    return matrix_1xk


def get_reduced_dimension_nxk_using_latent_semantics(all_data, all_latent_semantics, feature_model):
    reduced_matrix_nxk = []
    for each in all_data:
        reduced_matrix_nxk.append({feature_model: transform_1xm_to_1xk(each[feature_model], all_latent_semantics),
                                   'label': each['label']})

    return reduced_matrix_nxk
