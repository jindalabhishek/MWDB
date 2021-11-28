from Util.dao_util import DAOUtil
from VA_Files import *
from vector_util import convert_image_to_matrix
from Utils import *
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
        Executes Task 5
        Output Subject - latent semantics matrix, (subject-list of weight matrix)
    """
    train_path = input('Welcome to Task 5 Demo. Enter the training path: ')
    dimensions = int(input("Total reduced Dimensions: "))
    feature_model = input('Enter the feature model (CM, ELBP, HOG): ')
    number_of_bits = int(input('Enter the number of bits: '))
    query_image_path = input('Enter the path for query image: ')
    number_of_similar_images = int(input('Enter the t value for most similar images: '))

    reduce_flag = False
    if dimensions > 0:
        reduce_flag = True

    image_vector_matrix, all_labels = retrive_data(train_path, feature_model, dimensions, reduce_flag)
    image_labels = all_labels[3]
    approximations, partition_boundaries = VA_Approx(image_vector_matrix, number_of_bits)
    query_image_vector = convert_image_to_matrix(query_image_path)
    query_image_feature_descriptor = get_query_image_feature_descriptor(feature_model, query_image_vector)

    indexes_of_similar_images, n_buckets, n_objects = VA_SSA(image_vector_matrix, approximations, partition_boundaries,
                                                             query_image_feature_descriptor, number_of_similar_images,
                                                             True)

    knn = np.array([image_labels[int(i)] for i in indexes_of_similar_images])
    print("K nearest neighbors (sorted):\n" + str(knn))
    print("Number of buckets searched: " + str(n_buckets))
    print("Number of images considered: " + str(n_objects))

    # plt.scatter(image_vector_matrix[:, 0], image_vector_matrix[:, 1], color='blue', label='dataset')
    # plt.scatter(query_image_feature_descriptor[0], query_image_feature_descriptor[1], color='orange', label='Query')
    # plt.scatter(knn[:, 0], knn[:, 1], color='red', label='K NN to Q', marker='*')
    # plt.legend()
    # plt.show()


main()
