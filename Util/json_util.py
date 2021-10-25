import numpy
import json

from Phase_2.dimensionality_reduction.KMeans import KMeans
from Phase_2.dimensionality_reduction.LDA import LDA
from Phase_2.dimensionality_reduction.PCA import PCA
from Phase_2.dimensionality_reduction.SVD import SVD


class LatentSemanticFile:
    MODEL = "model"
    LATENT_FEATURES = "latent_features"
    TASK_OUTPUT = "task_output"
    REDUCTION_TECHNIQUE = "reduction_technique"
    TASK_ID = "task_id"

    def __init__(self,model, dimension_reduction, task_output,task_id):
        self.latent_features = []
        self.task_id = task_id
        self.model = model
        self.dimensionReduction = dimension_reduction
        self.task_output = task_output

    def serialize(self, outputPath):
        val = {LatentSemanticFile.MODEL: self.model,
               LatentSemanticFile.REDUCTION_TECHNIQUE: type(self.dimensionReduction).__name__,
               LatentSemanticFile.LATENT_FEATURES: self.dimensionReduction.serialize(),
               LatentSemanticFile.TASK_ID:self.task_id,
               LatentSemanticFile.TASK_OUTPUT: self.task_output}
        return json.dump(val, open(outputPath, "w"))

    @staticmethod
    def deserialize(inputPath):
        val = json.load(open(inputPath))
        dimensionReductionDict = {KMeans.__name__: KMeans,
                                  LDA.__name__: LDA,
                                  PCA.__name__: PCA,
                                  SVD.__name__: SVD}
        model_object = val[LatentSemanticFile.MODEL]
        reduction_technique = val[LatentSemanticFile.REDUCTION_TECHNIQUE]
        reduction_technique_object = dimensionReductionDict[reduction_technique]
        latent_features_object = val[LatentSemanticFile.LATENT_FEATURES]
        dimension_reduction_data = reduction_technique_object.deserialize(latent_features_object)
        task_output_data = val[LatentSemanticFile.TASK_OUTPUT]
        task_id = val.get(LatentSemanticFile.TASK_ID,None)
        return LatentSemanticFile(model_object, dimension_reduction_data, task_output_data,task_id)
