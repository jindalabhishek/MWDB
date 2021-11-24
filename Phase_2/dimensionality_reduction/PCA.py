import numpy as np
from scipy import linalg



class PCA:
    LATENT_FEATURES = "matrix_mxk"
    LATENT_FEATURE_POWERS = "matrix_kxk"
    INPUT_MATRIX = "matrix_nxm"
    OBJECTS_IN_K_DIMENSIONS = "matrix_nxk"

    def serialize(self):
        return {PCA.LATENT_FEATURES: self.latent_features.real.tolist(),
                PCA.LATENT_FEATURE_POWERS: self.power_val.real.tolist(),
                PCA.INPUT_MATRIX: self.input_matrix.tolist(),
                PCA.OBJECTS_IN_K_DIMENSIONS: self.objects_in_k_dimensions.tolist()}

    @staticmethod
    def deserialize(latent_feature_json_object):
        obj = PCA()
        obj.latent_features = np.array(latent_feature_json_object[PCA.LATENT_FEATURES])
        obj.power_val = np.array(latent_feature_json_object[PCA.LATENT_FEATURE_POWERS])
        obj.input_matrix = np.array(latent_feature_json_object[PCA.INPUT_MATRIX])
        obj.objects_in_k_dimensions = np.array(latent_feature_json_object[PCA.OBJECTS_IN_K_DIMENSIONS])
        return obj

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

    def __init__(self, cov_method='np') -> None:
        self.cov_method = cov_method

    def compute(self, data, k, *args):
        n_objects, n_features = data.shape
        COV = np.zeros((n_features, n_features))

        # Calculate COV matrix
        print("Calculating COV matrix")
        if self.cov_method == 'manual':
            for feature_i in range(n_features):
                feature_i_ave = np.mean(data[:, feature_i])
                for feature_j in range(n_features):
                    feature_j_ave = np.mean(data[:, feature_j])
                    sum = 0
                    for object in range(n_objects):
                        sum += (data[object, feature_i] - feature_i_ave) * (data[object, feature_j] - feature_j_ave)
                    COV[feature_i, feature_j] = sum / n_objects
        elif self.cov_method == 'auto':
            COV = np.dot(data.T, data) / (n_features - 1)  # Feature x Feature COV matrix -> Incorrect!
        elif self.cov_method == 'np':
            COV = np.cov(data, rowvar=False)

        # Find eigenvalues and eigenvectors
        print("Finding eigenvalues and eigenvectors")

        eig_val, eig_vec = PCA.top_eigen_vectors(*np.linalg.eig(COV), k)
        self.power_val = eig_val
        self.latent_features = eig_vec
        self.input_matrix = data
        self.objects_in_k_dimensions = np.matmul(data, eig_vec).real
        # Transform data
        # print("Transforming data")
        return self.objects_in_k_dimensions

    def transform(self, image_vector_matrix):

        try:
            matrix_nxk = np.matmul(image_vector_matrix, self.latent_features)
            return np.matmul(matrix_nxk, linalg.inv(np.diagflat(self.power_val)))
        except:
            return None
