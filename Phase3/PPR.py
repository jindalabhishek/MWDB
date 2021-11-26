import json
import os

import feature_descriptor_util

import Constants
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
from image_comparison_util import get_elbp_histogram


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

    labels = []
    for name in os.listdir(folder_1_path):
        labels.append(name)

    feature_model = 'color_moment'
    # TODO implement this function(assuming latent semantis preserves the order of the folder)
    training_latent_semantics = get_latent_semantics(folder_1_path, feature_model, k_value)

    # Looping each query and output the 'false positive and miss rates'.
    for name in os.listdir(folder_2_path):
        """
           Compute the image pixels for the image
        """
        image_pixels = vector_util.convert_image_to_matrix(folder_2_path + '/' + name)
        print('Image Size:', len(image_pixels), len(image_pixels[0]), 'Max Pixel Size:', np.amax(image_pixels))
        """
           Normalize the image pixels
        """
        image_pixels = image_pixels / Constants.GREY_SCALE_MAX
        """
            Compute All the feature descriptors
        """
        query_matrix_1xm = []

        if feature_model == 'color_moment':
            color_moment_feature_descriptor = feature_descriptor_util.get_color_moment_feature_descriptor(image_pixels)
            color_moment_feature_descriptor = feature_descriptor_util.get_reshaped_color_moment_vector(color_moment_feature_descriptor)
            query_matrix_1xm = color_moment_feature_descriptor.copy()
        elif feature_model == 'elbp':
            elbp_feature_descriptor = feature_descriptor_util.get_elbp_feature_descriptor(image_pixels)
            elbp_feature_descriptor = get_elbp_histogram(elbp_feature_descriptor)
            query_matrix_1xm = elbp_feature_descriptor.copy()
        elif feature_model == 'hog':
            hog_feature_descriptor = feature_descriptor_util.get_hog_feature_descriptor(image_pixels)
            query_matrix_1xm = hog_feature_descriptor.copy()

        query_matrix_1xm = np.array(query_matrix_1xm)
        query_matrix_1xk = np.matmul(query_matrix_1xm, np.array(training_latent_semantics['mxk']))
        query_matrix_1xk = np.array(query_matrix_1xk)
        # Find the Similarity list of query with every image in folder 1
        similarity_list = []        # it should be 1xn of float values.
        for each in training_latent_semantics['nxk']:
            similarity_list.append(np.linalg.norm(np.array(each), query_matrix_1xk))

        count_seeds = 3
        new_similarity_list = similarity_list.copy()
        seeds = new_similarity_list.sort(reverse=True)[:count_seeds]

        # TODO not considering 'n' as the limiting factor in no of relavant edges of an edge.
        # Initializing adjacency matrix.

        # TODO not sure if considering undirected edges will work for PPR.
        adjacency_matrix = np.empty(shape=(len(labels), len(labels)))
        adjacency_matrix.fill(0)

        # Adding edges  between same 'type' nodes.
        for i, label_i in enumerate(labels):
            for j, label_j in enumerate(labels):
                if i == j:
                    continue
                split_curr_label = label_i.split('-')
                type_i = split_curr_label[1]

                split_curr_label = label_j.split('-')
                type_j = split_curr_label[1]

                if type_i == type_j:
                    adjacency_matrix[]


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
