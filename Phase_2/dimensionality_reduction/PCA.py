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

    def compute(self,data,k,cov_method='np'):
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
        eig_val, eig_vec = np.linalg.eig(COV)

        if (k > eig_val.shape[0]):
            print("ERROR - k is larger than available latent semantics = " + str(eig_val.shape[0]))
            exit()

        # Sort eig_vec by eig_val in descending order
        print("Sorting and selecting " + str(k) + " latent semantics")
        sorted_eig_val = sorted(eig_val, reverse=True)
        eig_val_list = eig_val.tolist()
        sorted_eig_val_index = [eig_val_list.index(i) for i in sorted_eig_val]  # 0 is highest

        # Choose top-k latent features
        selected_eig_vec = []
        for i in range(k):
            selected_eig_vec.append(eig_vec[sorted_eig_val_index[i]])
        selected_eig_vec = np.transpose(np.array(selected_eig_vec))
        self.power_val = np.array(sorted_eig_val[:k])
        self.latent_features = selected_eig_vec
        # Transform data
        # print("Transforming data")
        return np.matmul(data, selected_eig_vec)

