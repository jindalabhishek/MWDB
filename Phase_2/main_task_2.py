import numpy

from Phase_2.dimensionality_reduction.KMeans import KMeans
from Phase_2.dimensionality_reduction.LDA import LDA
from Phase_2.dimensionality_reduction.PCA import PCA
from Phase_2.dimensionality_reduction.SVD import SVD
from Util.Utils import *
from Util.dao_util import DAOUtil
from numpy.linalg import svd
import numpy as np
from Util.json_util import LatentSemanticFile


def get_image_vector_matrix(feature_descriptors, feature_model):
    image_vector_matrix = []
    image_subjects = []
    for feature_descriptor in feature_descriptors:
        image_vector_matrix.append(feature_descriptor[feature_model])
        image_subjects.append(feature_descriptor['label'].split("-")[1])
    return image_vector_matrix, image_subjects


def main():
    """
        Executes Task 1
        Output type - latent semantics matrix, (type-list of weight matrix)
    """
    """
        Connection to MongoDB using PyMongo
    """
    dao_util = DAOUtil()
    feature_model = input('Welcome to Task 2 Demo. Enter the feature model (color_moment, elbp, hog):')
    feature_model_name = feature_model
    feature_model += '_feature_descriptor'
    subject_id = input('Enter Subject Id:')
    # subject_id = '1'
    k = int(input("Enter K Value for Dimensionality Reduction:"))
    dimension_reduction_technique = input('Select Dimension reduction technique: (1. PCA 2.SVD 3.LDA 4.k-means): ')

    feature_descriptors = dao_util.get_feature_descriptors_by_subject_id(subject_id)
    image_vector_matrix, image_subjects = get_image_vector_matrix(feature_descriptors, feature_model)

    # dimension_reduction_technique = '4'
    print('Image_vector_matrix dimension: ', len(image_vector_matrix), len(image_vector_matrix[0]))
    image_vector_matrix = numpy.array(image_vector_matrix)

    technique_number_vs_reduction_object = {'1': PCA(), '2': SVD(), '3': LDA(), '4': KMeans(1000)}
    dimension_reduction_object = technique_number_vs_reduction_object[dimension_reduction_technique]

    type_weight_matrix = dimension_reduction_object.compute(image_vector_matrix, k, image_subjects)
    type_weight_pairs = {}
    for i in range(len(image_subjects)):
        image_label = image_subjects[i]
        if image_label not in type_weight_pairs:
            type_weight_pairs[image_label] = []
        type_weight_pairs[image_label].append(type_weight_matrix[i])

    for i in type_weight_pairs:
        type_weight_pairs[i] = np.average(np.array(type_weight_pairs[i]), axis=0).tolist()

    sorted_type_weight_pairs = sort_feature_weight_pair(type_weight_pairs)
    save_task_data('task_2', dimension_reduction_object, task_output=sorted_type_weight_pairs, topic=subject_id,
                   feature_model=feature_model_name)
    # print('Entire type weight matrix: \n', type_weight_matrix)
    print_k_latent_semantics_in_sorted_weight_pairs(sorted_type_weight_pairs)


main()
