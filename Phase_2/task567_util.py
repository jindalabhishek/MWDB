from matrix_util import *
import numpy as np
matrix_mxk = None
matrix_kxk = None
matrix_nxm = None
matrix_nxk = None


def transform_1xm_to_1xk(matrix_1xm, all_latent_semantics, task_number,matrix_mxk=None,matrix_kxk=None,matrix_nxm=None,matrix_nxk = None):

    matrix_1xk = []
    method_dimension_reduction = all_latent_semantics['reduction_technique']
    latent_semantics = all_latent_semantics['latent_features']

    # setting global variables

    if method_dimension_reduction == 'PCA':
        if task_number == 'task_1' or task_number == 'task_2':
            matrix_mxk = latent_semantics['matrix_mxk']
        matrix_kxk = latent_semantics['matrix_kxk']
        matrix_1xk = multiply_matrices(matrix_1xm, matrix_mxk)
        matrix_1xk = multiply_matrices(matrix_1xk, inverse_matrix(np.diagflat(matrix_kxk)))

    elif method_dimension_reduction == 'SVD':
        if task_number == 'task_1' or task_number == 'task_2':
            matrix_mxk = latent_semantics['matrix_mxk']
            matrix_kxk = latent_semantics['matrix_kxk']
        matrix_1xk = multiply_matrices(matrix_1xm, matrix_mxk)
        matrix_1xk = multiply_matrices(matrix_1xk, inverse_matrix(np.diagflat(matrix_kxk)))

    elif method_dimension_reduction == 'LDA':
        matrix_nxm = latent_semantics['matrix_nxm']
        if task_number == 'task_1' or task_number == 'task_2':
            matrix_nxk = latent_semantics['matrix_nxk']
        matrix_mxk = multiply_matrices(transpose_matrix(matrix_nxm), matrix_nxk)
        matrix_1xk = multiply_matrices(matrix_1xm, matrix_mxk)

    elif method_dimension_reduction == 'KMeans':
        if task_number == 'task_1' or task_number == 'task_2':
            matrix_kxm = latent_semantics['centroids_kxm']
            # Euclidean distance between query and all centroids
            for each in matrix_kxm:
                matrix_1xk.append(np.linalg.norm(np.array(matrix_1xm) - np.array(each)))
        elif task_number == 'task_3':
            matrix_kxt = latent_semantics['centroids_kxm']
            # Euclidean distance between query and all centroids
            for each in matrix_kxt:
                matrix_1xk.append(np.linalg.norm(np.array(matrix_1xm) - np.array(each)))

    return matrix_1xk

def getTransformMatrix(all_data,feature_model,task_number):
    multiplier = None
    idx = 1 if task_number=='task_3' else 2
    if task_number=='task_3' or task_number == 'task_4':
        all_data.sort(key=lambda x: x['label'].split('-')[idx])
        curr_label = all_data[0]['label'].split('-')[idx]
        curr_temp_list = []
        averaged_nxm_to_sxm = []
        for each in all_data:
            if curr_label == each['label'].split('-')[idx]:
                curr_temp_list.append(each[feature_model])
            else:
                averaged_nxm_to_sxm.append(np.mean(np.array(curr_temp_list), axis=0).tolist())
                curr_label = each['label'].split('-')[idx]
                curr_temp_list = []
        averaged_nxm_to_sxm.append(np.mean(np.array(curr_temp_list), axis=0).tolist())
        multiplier = transpose_matrix(averaged_nxm_to_sxm)
    return multiplier

def get_reduced_dimension_nxk_using_latent_semantics(all_data, dimension_reduction, feature_model, multiplier=None,flag=None):
    reduced_matrix_nxk = []
    for each in all_data:
        reduced_matrix_nxk.append({feature_model: dimension_reduction.transform(np.matmul(np.array(each[feature_model]),multiplier) if flag else np.array(each[feature_model])),
                                   'label': each['label']})
    return reduced_matrix_nxk

