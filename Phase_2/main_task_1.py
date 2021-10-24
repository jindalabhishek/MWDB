import numpy

from Phase_2.dimensionality_reduction.KMeans import KMeans
from Phase_2.dimensionality_reduction.LDA import LDA
from Phase_2.dimensionality_reduction.PCA import PCA
from Phase_2.dimensionality_reduction.SVD import SVD
from Util.dao_util import DAOUtil
import numpy as np
from Util.Utils import save_task_data, sort_feature_weight_pair


def get_image_vector_matrix(feature_descriptors, feature_model):
    image_vector_matrix = []
    image_types = []
    for feature_descriptor in feature_descriptors:
        image_vector_matrix.append(feature_descriptor[feature_model])
        image_types.append(feature_descriptor['label'].split("-")[2])
    return image_vector_matrix, image_types


def main():
    """
        Executes Task 1
        Output Subject - latent semantics matrix, (subject-list of weight matrix)
    """
    """
        Connection to MongoDB using PyMongo
    """
    dao_util = DAOUtil()
    feature_model = input('Welcome to Task 1 Demo. Enter the feature model (color_moment, elbp, hog):')

    # feature_model = 'hog'
    feature_model_name = feature_model
    feature_model += '_feature_descriptor'

    type_id = input('Enter Type Id:')
    # type_id = 'cc'
    feature_descriptors = dao_util.get_feature_descriptors_by_type_id(type_id)
    image_vector_matrix, image_types = get_image_vector_matrix(feature_descriptors, feature_model)
    dimension_reduction_technique = input('Select Dimension reduction technique: (1. PCA 2.SVD 3.LDA 4.k-means): ')
    # dimension_reduction_technique = '1'
    print('Image_vector_matrix dimension: ', len(image_vector_matrix), len(image_vector_matrix[0]))
    image_vector_matrix = numpy.array(image_vector_matrix)
    k = int(input("Enter K Value for Dimensionality Reduction:"))
    technique_number_vs_reduction_object = {'1': PCA(), '2': SVD(), '3': LDA(), '4': KMeans(1000)}
    dimension_reduction_object = technique_number_vs_reduction_object[dimension_reduction_technique]
    subject_weight_matrix = dimension_reduction_object.compute(image_vector_matrix, k, image_types)

    subject_weight_pairs = {}
    for i in range(len(image_types)):
        image_label = image_types[i]
        if image_label not in subject_weight_pairs:
            subject_weight_pairs[image_label] = []
        subject_weight_pairs[image_label].append(subject_weight_matrix[i])

    for i in subject_weight_pairs:
        subject_weight_pairs[i] = np.average(np.array(subject_weight_pairs[i]), axis=0).tolist()

    save_task_data('task_1', dimension_reduction_object, task_output=sort_feature_weight_pair(subject_weight_pairs), topic=type_id,
                   feature_model=feature_model_name)
    # print('Subject_weight_matrix dimension', len(subject_weight_matrix), len(subject_weight_matrix[0]))
    print('Entire Subject weight matrix: \n', subject_weight_matrix)


main()
