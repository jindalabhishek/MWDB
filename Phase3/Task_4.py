import numpy as np

from Util.dao_util import DAOUtil
from LSH import *
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

    q_filename = query_image_path.split('/')[-1]
    q_labels = q_filename[:-4].split('-')[1:]
    print(q_labels)
    # feature_model = 'hog'
    feature_model_name = feature_model
    feature_model += '_feature_descriptor'

    feature_descriptors = dao_util.get_feature_descriptors_for_all_images()
    image_vector_matrix, image_labels = get_image_vector_matrix(feature_descriptors, feature_model)

    LSH = LSHash(np.array(image_vector_matrix).shape[1], number_of_bits, number_of_families)

    for inp,label in zip(image_vector_matrix, image_labels):
      LSH.index(inp, label)

    query_image_vector = convert_image_to_matrix(query_image_path)
    query_image_feature_descriptor = get_query_image_feature_descriptor(feature_model_name, query_image_vector)

    if number_of_similar_images == 0:
        planes,dist ,k= LSH.query(query_image_feature_descriptor, None)
    else:
        planes,k = LSH.query(query_image_feature_descriptor, number_of_similar_images)

    if  not(isinstance(image_labels,list)):
      image_labels=image_labels.tolist()
    if  not(isinstance(image_vector_matrix,list)):
      image_vector_matrix = image_vector_matrix.tolist()

    labels_of_similar_images = []
    for x in planes:
      labels_of_similar_images.append(image_labels[image_vector_matrix.index(x.tolist())])

    knn = np.array(labels_of_similar_images)
    print("K nearest neighbors (sorted):\n" + str(knn))
    type_miss=0
    subject_miss = 0
    id_miss = 0
    for name in knn:
        ll = name[:-4].split('-')[1:]
        if ll[0] != q_labels[0]:
            print(ll[0] ,q_labels[0])
            type_miss += 1
        if ll[1] != q_labels[1]:
            subject_miss += 1
        if ll[2] != q_labels[2]:
            id_miss += 1
    print('--type FP--'+str(k-130)+'--subject FP--'+str(k-400)+'--ID FP--'+str(k-(13*40)))
    print('--type miss--' + str(type_miss) + '--subject miss--' + str(subject_miss) + '--ID miss--' + str(id_miss))

    # plt.scatter(image_vector_matrix[:, 0], image_vector_matrix[:, 1], color='blue', label='dataset')
    # plt.scatter(query_image_feature_descriptor[0], query_image_feature_descriptor[1], color='orange', label='Query')
    # plt.scatter(knn[:, 0], knn[:, 1], color='red', label='K NN to Q', marker='*')
    # plt.legend()
    # plt.show()


main()
