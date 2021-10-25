import numpy as np
import sklearn.decomposition as sk_decomp


class LDA:
    LATENT_FEATURES = "matrix_nxk"
    INPUT_MATRIX = "matrix_nxm"

    def serialize(self):
        return {LDA.LATENT_FEATURES: self.latent_features.real.tolist(),
                LDA.INPUT_MATRIX: self.input_matrix.tolist()}

    @staticmethod
    def deserialize(latent_feature_json_object):
        obj = LDA()
        obj.latent_features = np.array(latent_feature_json_object[LDA.LATENT_FEATURES])
        obj.input_matrix = np.array(latent_feature_json_object[LDA.INPUT_MATRIX])
        return obj

    @staticmethod
    def normalize_data_for_lda(image_vector_matrix):
        normalized_data = (image_vector_matrix - np.min(image_vector_matrix)) \
                          / (np.max(image_vector_matrix) - np.min(image_vector_matrix))
        return normalized_data

    def compute(self, data, k, image_types, *args):
        lda = sk_decomp.LatentDirichletAllocation(n_components=k, random_state=0)
        latent_features = lda.fit_transform(LDA.normalize_data_for_lda(data), image_types)
        self.latent_features = latent_features
        self.input_matrix = data
        return latent_features.real

    def transform(self, image_vector_matrix):
        matrix_mxk = np.matmul(self.input_matrix.transpose(), self.latent_features)
        return np.matmul(image_vector_matrix, matrix_mxk)

