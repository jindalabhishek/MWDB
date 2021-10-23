import numpy as np
import sklearn.decomposition as sk_decomp


class LDA:
    LATENT_FEATURES = "features_in_k_dimensions"
    LATENT_FEATURE_POWERS = "powers"

    def serialize(self):
        return {LDA.LATENT_FEATURES:self.latent_features.real.tolist()}

    @staticmethod
    def deserialize(dict):
        obj = LDA()
        obj.latent_features = np.array(dict[LDA.LATENT_FEATURES])
        obj.power_val = np.array(dict[LDA.LATENT_FEATURE_POWERS])
        return obj

    def compute(self,data,k):
        lda = sk_decomp.LatentDirichletAllocation(n_components=k, random_state=0)
        latent_features = lda.fit_transform(data)
        self.latent_features = latent_features
        return latent_features.real
