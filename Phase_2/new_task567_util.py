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
        matrix_mxk = multiply_matrices(transpose_matrix(latent_semantics['matrix_nxm']), latent_semantics['matrix_nxk'])
        matrix_1xk = multiply_matrices(matrix_1xm, matrix_mxk)
    elif method_dimension_reduction == 'KMeans':
        matrix_kxm = latent_semantics['centroids_kxm']
        # Euclidean distance between query and all centroids
        for each in matrix_kxm:
            matrix_1xk.append(np.linalg.norm(np.array(matrix_1xm) - np.array(each)))

    return matrix_1xk


def get_reduced_dimension_nxk_using_latent_semantics(all_data, dimension_reduction, feature_model,):
    reduced_matrix_nxk = []
    for each in all_data:
        reduced_matrix_nxk.append({feature_model: dimension_reduction.transform(np.array(each[feature_model])),
                                   'label': each['label']})

    return reduced_matrix_nxk