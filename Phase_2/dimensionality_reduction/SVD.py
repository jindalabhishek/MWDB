import numpy as np
from scipy import linalg

from Phase_2.dimensionality_reduction.PCA import PCA


class SVD:
    LATENT_FEATURES = "matrix_mxk"
    LATENT_FEATURE_POWERS = "matrix_kxk"
    INPUT_MATRIX = "matrix_nxm"
    OBJECTS_IN_K_DIMENSIONS = "matrix_nxk"

    def compute(self, image_vector_matrix, k, *args):
        right_eig_val, right_eig_vec = PCA.top_eigen_vectors(
            *np.linalg.eig(np.dot(image_vector_matrix.T, image_vector_matrix)), k)
        left_eig_val, left_eig_vec = PCA.top_eigen_vectors(
            *np.linalg.eig(np.dot(image_vector_matrix, image_vector_matrix.T)), k)
        self.power_val = right_eig_val
        self.latent_features = right_eig_vec
        self.input_matrix = image_vector_matrix
        self.objects_in_k_dimensions = left_eig_vec.real
        return self.objects_in_k_dimensions

    def serialize(self):
        return {SVD.LATENT_FEATURES: self.latent_features.real.tolist(),
                SVD.LATENT_FEATURE_POWERS: self.power_val.real.tolist(),
                SVD.INPUT_MATRIX: self.input_matrix.tolist(),
                SVD.OBJECTS_IN_K_DIMENSIONS: self.objects_in_k_dimensions.tolist()}

    @staticmethod
    def deserialize(latent_feature_json_object):
        obj = SVD()
        obj.latent_features = np.array(latent_feature_json_object[SVD.LATENT_FEATURES])
        obj.power_val = np.array(latent_feature_json_object[SVD.LATENT_FEATURE_POWERS])
        obj.input_matrix = np.array(latent_feature_json_object[SVD.INPUT_MATRIX])
        obj.objects_in_k_dimensions = np.array(latent_feature_json_object[SVD.OBJECTS_IN_K_DIMENSIONS])
        return obj


    def transform(self,image_vector_matrix):
        matrix_nxk = np.matmul(image_vector_matrix, self.latent_features)
        return np.matmul(matrix_nxk, linalg.inv(np.diagflat(self.power_val)))
