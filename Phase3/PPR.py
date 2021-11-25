import json
import os

import vector_util
from Util.dao_util import DAOUtil
from sklearn.decomposition import LatentDirichletAllocation
from Util.k_means_util import reduce_dimensions_k_means
from Util.graph_util import *
import numpy as np
from Util.json_util import LatentSemanticFile
from dimensionality_reduction import LDA
from Util.Utils import *

# def get_image_vector_matrix(feature_descriptors, feature_model):
#     image_vector_matrix = []
#     for feature_descriptor in feature_descriptors:
#         image_vector_matrix.append(feature_descriptor[feature_model])
#     return image_vector_matrix
#
#
# def get_image_label_array(feature_descriptors):
#     image_label_array = []
#     for feature_descriptor in feature_descriptors:
#         label = feature_descriptor['label']
#         image_type = label.split('-')[1]
#         image_label_array.append(image_type)
#         # image_label_array.append(i)
#     return image_label_array
#
#
# def calculate_lda(image_vector_matrix, image_types, components):
#     lda = LatentDirichletAllocation(n_components=components)
#     image_vector_matrix_lda = lda.fit_transform(image_vector_matrix, image_types)
#     return image_vector_matrix_lda
#
#
# def normalize_data_for_lda(image_vector_matrix):
#     normalized_data = (image_vector_matrix - np.min(image_vector_matrix)) \
#                       / (np.max(image_vector_matrix) - np.min(image_vector_matrix))
#     return normalized_data



def main():
    """
        Executes Task 9
        Output Subject - latent semantics matrix, (subject-list of weight matrix)
    """
    """
        Connection to MongoDB using PyMongo
    """
    # 1 for type, 2 for subject and 3 for sample
    # assuming task 1 for now. => type.

    # create type-type from folder 1.
    folder_1_path = input('Enter folder 1 path: ')
    folder_2_path = input('Enter folder 2 path: ')

    similarity_matrix_txt = create_similarity_matrix_from_images(folder_1_path)
    similarity_matrix = np.array(similarity_matrix_txt)

    parent_dir = folder_1_path.split('\\')
    parent_dir = parent_dir[len(parent_dir) - 1]
    print('Base Path:', parent_dir)
    for name in os.listdir(folder_2_path):
        """
           Compute the image pixels for the image
        """
        image_pixels = vector_util.convert_image_to_matrix(folder_path + '\\' + name)
        print('Image Size:', len(image_pixels), len(image_pixels[0]), 'Max Pixel Size:', np.amax(image_pixels))
        """
           Normalize the image pixels
        """
        image_pixels = image_pixels / Constants.GREY_SCALE_MAX
        """
            Compute All the feature descriptors
        """
        color_moment_feature_descriptor = feature_descriptor_util.get_color_moment_feature_descriptor(image_pixels)
        color_moment_feature_descriptor = feature_descriptor_util \
            .get_reshaped_color_moment_vector(color_moment_feature_descriptor)
    n = int(input('Enter value of n:'))
    subject_ids = set()
    subject_ids.add(subject_id_1)
    subject_ids.add(subject_id_2)
    subject_ids.add(subject_id_3)

    adjacency_matrix = convert_similarity_matrix_to_graph(similarity_matrix, n)
    hubs_vs_authorities = get_hubs_authorities_from_adjacency_matrix(adjacency_matrix)
    transition_matrix = get_transition_matrix_from_hubs_authorities(hubs_vs_authorities)
    seed_nodes = get_seed_nodes(subject_ids, len(similarity_matrix))
    ppr_matrix = get_page_ranking(0.4, transition_matrix, seed_nodes)
    print(ppr_matrix)
    highest_subject_ids = np.argsort(-ppr_matrix[:, -1])
    print(highest_subject_ids)
    print('Most Relevant M Subjects Ids w.r.t to seed nodes')
    print(highest_subject_ids[:m]+1)


main()
