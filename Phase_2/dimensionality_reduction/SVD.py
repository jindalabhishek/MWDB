import numpy as np
class SVD:
    LATENT_FEATURES = "latent_features"
    LATENT_FEATURE_POWERS = "powers"

    @staticmethod
    def top_eigen_vectors(eigen_value, eigen_vec, k):
        val = []
        bool_arr = np.isreal(eigen_value)
        for i in range(len(eigen_value)):
            if bool_arr[i] and eigen_value[i] >= 0:
                val.append((eigen_value[i], i))
        eigen_value = sorted(val, reverse=True)[:k]
        eigen_vec = eigen_vec.T
        eigen_vec = np.array([eigen_vec[i[1]] for i in eigen_value])
        eigen_vec = eigen_vec.T
        return np.array([i[0] for i in eigen_value]), eigen_vec

    def compute(self,image_vector_matrix, k):
        right_eig_val, right_eig_vec = SVD.top_eigen_vectors(
            *np.linalg.eig(np.dot(image_vector_matrix.T, image_vector_matrix)), k)
        left_eig_val, left_eig_vec = SVD.top_eigen_vectors(
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



