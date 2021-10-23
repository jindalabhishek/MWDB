
FEATURE_POWER = "feature_power"
FEATURE = "feature"
class PCAWriter:

    def serialize(self,latent_feature):
        return {FEATURE:latent_feature[0],FEATURE_POWER:latent_feature[1]}

