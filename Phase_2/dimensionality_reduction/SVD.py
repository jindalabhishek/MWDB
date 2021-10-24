import numpy as np

from Phase_2.dimensionality_reduction.PCA import PCA


class SVD:
    LATENT_FEATURES = "latent_features"
    LATENT_FEATURE_POWERS = "powers"

    def compute(self,image_vector_matrix, k,*args):
        right_eig_val, right_eig_vec = PCA.top_eigen_vectors(
            *np.linalg.eig(np.dot(image_vector_matrix.T, image_vector_matrix)), k)
        left_eig_val, left_eig_vec = PCA.top_eigen_vectors(
            *np.linalg.eig(np.dot(image_vector_matrix, image_vector_matrix.T)), k)
        self.power_val = right_eig_val
        self.latent_features = right_eig_vec
        return left_eig_vec.real

    def serialize(self):
        return {SVD.LATENT_FEATURES:self.latent_features.real.tolist(),SVD.LATENT_FEATURE_POWERS:self.power_val.real.tolist()}

    @staticmethod
    def deserialize(dict):
        obj = SVD()
        obj.latent_features = np.array(dict[SVD.LATENT_FEATURES])
        obj.power_val = np.array(dict[SVD.LATENT_FEATURE_POWERS])
        return obj



