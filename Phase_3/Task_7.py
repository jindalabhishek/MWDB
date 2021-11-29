from vector_util import convert_image_to_matrix
from Utils import *
from Feedback_SVM import *
from file import *


def get_image_vector_matrix(feature_descriptors, feature_model):
    image_vector_matrix = []
    image_labels = []
    for feature_descriptor in feature_descriptors:
        image_vector_matrix.append(feature_descriptor[feature_model])
        image_labels.append(feature_descriptor['label'])
    return image_vector_matrix, image_labels


def main():
    """
        Executes Task 7
    """
    train_path = input('Welcome to Task 7 Demo. Enter the training path: ')
    dimensions = int(input("Total reduced Dimensions: "))
    feature_model = input('Enter the feature model (CM, ELBP, HOG): ')
    number_of_bits = int(input('Enter the number of bits:'))
    query_image_path = input('Enter the path for query image:')
    number_of_similar_images = int(input('Enter the t value for most similar images:'))

    query_image_vector = convert_image_to_matrix(query_image_path)
    query_image_feature_descriptor = get_query_image_feature_descriptor(feature_model, query_image_vector)

    if dimensions > 0:
        dimension_reduction, image_labels = getTrainData(train_path, feature_model, dimensions, getType)
        image_vector_matrix = dimension_reduction.objects_in_k_dimensions
        query_image_feature_descriptor = dimension_reduction.transform(query_image_feature_descriptor)

    else:
        image_vector_matrix, all_types, image_labels = getImageData(train_path, feature_model, getType)

    SVM_RF(image_vector_matrix, image_labels, query_image_feature_descriptor, number_of_bits, number_of_similar_images)


# main()
