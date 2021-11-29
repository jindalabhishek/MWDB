from Util.dao_util import DAOUtil
from VA_Files import *
from vector_util import convert_image_to_matrix
from Utils import *
from file import *
from sklearn.neighbors import KNeighborsClassifier


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
    query_image_vector = convert_image_to_matrix(query_image_path)
    query_image_feature_descriptor = get_query_image_feature_descriptor(feature_model, query_image_vector)
    if dimensions > 0:
        dimension_reduction, image_labels = getTrainData(train_path, feature_model, dimensions, getType)
        image_vector_matrix = dimension_reduction.objects_in_k_dimensions
        query_image_feature_descriptor = dimension_reduction.transform(query_image_feature_descriptor)

    else:
        image_vector_matrix, all_types, image_labels = getImageData(train_path, feature_model, getType)

    approximations, partition_boundaries = VA_Approx(image_vector_matrix, number_of_bits)

    indexes_of_similar_images, n_buckets, n_objects = VA_SSA(image_vector_matrix, approximations, partition_boundaries,
                                                             query_image_feature_descriptor, number_of_similar_images,
                                                             True)

    knn = np.array([image_labels[int(i)] for i in indexes_of_similar_images])
    print("K nearest neighbors (sorted):\n" + str(knn))
    print("Number of buckets searched: " + str(n_buckets))
    print("Number of images considered: " + str(n_objects))

    distances = np.zeros(len(image_vector_matrix))
    for i in range(0, len(image_vector_matrix)):
        distances[i] = np.linalg.norm(image_vector_matrix[i]-query_image_feature_descriptor)
    indexes = np.argsort(distances)
    correct_images = [image_labels[index] for index in indexes]
    correct_images = correct_images[:number_of_similar_images]
    print(correct_images)

    fp_rate = 0
    miss_rate = 0
    for label in knn:
        if label not in correct_images:
            fp_rate += 1
    fp_rate = fp_rate/len(knn)

    for label in correct_images:
        if label not in knn:
            miss_rate += 1
    miss_rate = miss_rate/len(correct_images)

    print('False Positive Rate: ', fp_rate)
    print('Miss Rate: ', miss_rate)
    # types = [n_objects.split('-')[1]]
    # plt.scatter(image_vector_matrix[:, 0], image_vector_matrix[:, 1], color='blue', label='dataset')
    # plt.scatter(query_image_feature_descriptor[0], query_image_feature_descriptor[1], color='orange', label='Query')
    # plt.scatter(knn[:, 0], knn[:, 1], color='red', label='K NN to Q', marker='*')
    # plt.legend()
    # plt.show()


main()
