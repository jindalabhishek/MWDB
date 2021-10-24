import numpy as np
import sklearn.decomposition as sk_decomp


class LDA:
    LATENT_FEATURES = "latent_features"
    LATENT_FEATURE_POWERS = "powers"

    def serialize(self):
        return {LDA.LATENT_FEATURES:self.latent_features.real.tolist()}

    @staticmethod
    def deserialize(dict):
        obj = LDA()
        obj.latent_features = np.array(dict[LDA.LATENT_FEATURES])
        obj.power_val = np.array(dict[LDA.LATENT_FEATURE_POWERS])
        return obj

    @staticmethod
    def normalize_data_for_lda(image_vector_matrix):
        normalized_data = (image_vector_matrix - np.min(image_vector_matrix)) \
                          / (np.max(image_vector_matrix) - np.min(image_vector_matrix))
        return normalized_data

    def compute(self,data,k,imageTypes,*args):
        lda = sk_decomp.LatentDirichletAllocation(n_components=k, random_state=0)
        latent_features = lda.fit_transform(LDA.normalize_data_for_lda(data),imageTypes)
        self.latent_features = latent_features
        return latent_features.real
