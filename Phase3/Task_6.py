from Util.dao_util import DAOUtil
from VA_Files import *
from vector_util import convert_image_to_matrix
from Utils import *
from Feedback_DT import *
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
        Executes Task 6
    """
    train_path = input('Welcome to Task 6 Demo. Enter the training path: ')
    dimensions = int(input("Total reduced Dimensions: "))
    feature_model = input('Enter the feature model (CM, ELBP, HOG): ')
    query_image_path = input('Enter the path for query image:')
    number_of_similar_images = int(input('Enter the t value for most similar images:'))
    # query_image_path= '/home/kirity/Downloads/4000/image-cc-7-3.png'

    reduce_flag = False
    if dimensions > 0:
        reduce_flag = True

    image_vector_matrix, all_labels = retrive_data(train_path, feature_model, dimensions, reduce_flag)
    image_labels = all_labels[3]

    query_image_vector = convert_image_to_matrix(query_image_path)
    query_image_feature_descriptor = get_query_image_feature_descriptor(feature_model, query_image_vector)

    indexes_of_similar_images = DT_RF(np.array(image_vector_matrix), image_labels, query_image_feature_descriptor,
                                      number_of_similar_images)


main()
