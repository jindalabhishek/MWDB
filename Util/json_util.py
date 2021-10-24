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

    def __init__(self, model, dimensionRecution, task_output):
        self.latent_features = []
        self.model = model
        self.dimensionReduction = dimensionRecution
        self.task_output = task_output

    def serialize(self, outputPath):
        val = {LatentSemanticFile.MODEL: self.model,
               LatentSemanticFile.REDUCTION_TECHNIQUE: type(self.dimensionReduction).__name__,
               LatentSemanticFile.LATENT_FEATURES: self.dimensionReduction.serialize(),
               LatentSemanticFile.TASK_OUTPUT: self.task_output}
        print(val)
        return json.dump(val, open(outputPath, "w"))

    @staticmethod
    def deserialize(inputPath):
        val = json.load(open(inputPath))
        dimensionReductionDict = {KMeans.__name__: KMeans, LDA.__name__: LDA, PCA.__name__: PCA, SVD.__name__: SVD}
        return LatentSemanticFile(val[LatentSemanticFile.MODEL],
                                  dimensionReductionDict[val[LatentSemanticFile.REDUCTION_TECHNIQUE]].deserialize(
                                      val[LatentSemanticFile.LATENT_FEATURES]),
                                  val[LatentSemanticFile.TASK_OUTPUT])
