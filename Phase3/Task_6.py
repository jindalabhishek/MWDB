from Util.dao_util import DAOUtil
from VA_Files import *
from vector_util import convert_image_to_matrix
from Utils import *
from Feedback_DT import *


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
    """
        Connection to MongoDB using PyMongo
    """
    dao_util = DAOUtil()
    feature_model = input('Welcome to Task 6 Demo. Enter the feature model (color_moment, elbp, hog):')
    query_image_path = input('Enter the path for query image:')
    number_of_similar_images = int(input('Enter the t value for most similar images:'))
    # query_image_path= '/home/kirity/Downloads/4000/image-cc-7-3.png'
    # feature_model = 'hog'1
    feature_model_name = feature_model
    feature_model += '_feature_descriptor'

    feature_descriptors = dao_util.get_feature_descriptors_for_all_images()
    image_vector_matrix, image_labels = get_image_vector_matrix(feature_descriptors, feature_model)
    query_image_vector = convert_image_to_matrix(query_image_path)
    query_image_feature_descriptor = get_query_image_feature_descriptor(feature_model_name, query_image_vector)

    indexes_of_similar_images = DT_RF(np.array(image_vector_matrix), image_labels, query_image_feature_descriptor,
                                      number_of_similar_images)


main()
