import numpy as np

class PCA:
    LATENT_FEATURES = "latent_features"
    LATENT_FEATURE_POWERS = "powers"

    def serialize(self):
        return {PCA.LATENT_FEATURES:self.latent_features.real.tolist(),PCA.LATENT_FEATURE_POWERS:self.power_val.real.tolist()}

    @staticmethod
    def deserialize(dict):
        obj = PCA()
        obj.latent_features = np.array(dict[PCA.LATENT_FEATURES])
        obj.power_val = np.array(dict[PCA.LATENT_FEATURE_POWERS])
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

    def compute(self,data,k,cov_method='np',*args):
        n_objects, n_features = data.shape
        COV = np.zeros((n_features, n_features))

        # Calculate COV matrix
        print("Calculating COV matrix")
        if (cov_method == 'manual'):
            for feature_i in range(n_features):
                feature_i_ave = np.mean(data[:, feature_i])
                for feature_j in range(n_features):
                    feature_j_ave = np.mean(data[:, feature_j])
                    sum = 0
                    for object in range(n_objects):
                        sum += (data[object, feature_i] - feature_i_ave) * (data[object, feature_j] - feature_j_ave)
                    COV[feature_i, feature_j] = sum / n_objects
        elif (cov_method == 'auto'):
            COV = np.dot(data.T, data) / (n_features - 1)  # Feature x Feature COV matrix -> Incorrect!
        elif (cov_method == 'np'):
            COV = np.cov(data, rowvar=False)

        # Find eigenvalues and eigenvectors
        print("Finding eigenvalues and eigenvectors")

        eig_val, eig_vec = PCA.top_eigen_vectors(*np.linalg.eig(COV),k)
        self.power_val = eig_val
        self.latent_features = eig_vec
        # Transform data
        # print("Transforming data")
        return np.matmul(data, eig_vec).real
