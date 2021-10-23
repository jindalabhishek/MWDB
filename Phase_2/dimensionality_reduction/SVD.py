import numpy as np

from Phase_2.dimensionality_reduction.PCA import PCA


class SVD:
    LATENT_FEATURES = "matrix_mxk"
    LATENT_FEATURE_POWERS = "matrix_kxk"

    def compute(self,image_vector_matrix, k):
        right_eig_val, right_eig_vec = PCA.top_eigen_vectors(
            *np.linalg.eig(np.dot(image_vector_matrix.T, image_vector_matrix)), k)
        left_eig_val, left_eig_vec = PCA.top_eigen_vectors(
            *np.linalg.eig(np.dot(image_vector_matrix, image_vector_matrix.T)), k)
        self.power_val = right_eig_val
        self.latent_features = right_eig_vec
        return left_eig_vec

    def serialize(self):
        return {SVD.LATENT_FEATURES:self.latent_features.real.tolist(),SVD.LATENT_FEATURE_POWERS:self.power_val.real.tolist()}

    @staticmethod
    def deserialize(dict):
        obj = SVD()
        obj.latent_features = np.array(dict[SVD.LATENT_FEATURES])
        obj.power_val = np.array(dict[SVD.LATENT_FEATURE_POWERS])
        return obj



