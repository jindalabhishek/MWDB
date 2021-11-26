
from Util.dao_util import DAOUtil
from LSH import *
from vector_util import convert_image_to_matrix
from Utils import *


def get_image_vector_matrix(feature_descriptors, feature_model):
    image_vector_matrix = []
    image_labels = []
    for feature_descriptor in feature_descriptors:
        image_vector_matrix.append(feature_descriptor[feature_model])
        image_labels.append(feature_descriptor['label'])
    return image_vector_matrix, image_labels


def main():
    """
        Executes Task 1
        Output Subject - latent semantics matrix, (subject-list of weight matrix)
    """
    """
        Connection to MongoDB using PyMongo
    """
    dao_util = DAOUtil()
    feature_model = input('Welcome to Task 4 Demo. Enter the feature model (color_moment, elbp, hog): ')
    number_of_bits = int(input('Enter the number of hash functions in a family of hash functions: '))
    number_of_families = int(input('Enter the number of hash families: '))
    query_image_path = input('Enter the path for query image:')
    number_of_similar_images = int(input('Enter the t value for most similar images:'))

    # feature_model = 'hog'
    feature_model_name = feature_model
    feature_model += '_feature_descriptor'

    feature_descriptors = dao_util.get_feature_descriptors_for_all_images()
    image_vector_matrix, image_labels = get_image_vector_matrix(feature_descriptors, feature_model)

    LSH = LSHash(image_vector_matrix.shape[1], number_of_bits, number_of_families)

    for inp in image_vector_matrix:
      LSH.index(inp)

    query_image_vector = convert_image_to_matrix(query_image_path)
    query_image_feature_descriptor = get_query_image_feature_descriptor(feature_model_name, query_image_vector)

    planes = LSH.query(query_image_feature_descriptor, number_of_similar_images)

    if  not(isinstance(image_labels,list)):
      image_labels=image_labels.tolist()
    if  not(isinstance(image_vector_matrix,list)):
      image_vector_matrix = image_vector_matrix.tolist()

    labels_of_similar_images = []
    for x in planes:
      labels_of_similar_images.append(image_labels[image_vector_matrix.index(x.tolist())])

    knn = np.array(labels_of_similar_images)
    print("K nearest neighbors (sorted):\n" + str(knn))

    # plt.scatter(image_vector_matrix[:, 0], image_vector_matrix[:, 1], color='blue', label='dataset')
    # plt.scatter(query_image_feature_descriptor[0], query_image_feature_descriptor[1], color='orange', label='Query')
    # plt.scatter(knn[:, 0], knn[:, 1], color='red', label='K NN to Q', marker='*')
    # plt.legend()
    # plt.show()


main()
