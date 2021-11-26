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

def getAdjecencyMatrix(labels):
    adjacency_matrix = np.empty(shape=(len(labels), len(labels)))
    adjacency_matrix.fill(0)
    for i, label_i in enumerate(labels):
        for j, label_j in enumerate(labels):
            if i == j:
                continue

            if label_j == label_i:
                adjacency_matrix[i][j] = 1
                adjacency_matrix[j][i] = 1
    return adjacency_matrix

def getTestingLabels(training_latent_semantics,train_labels,testing_latent_semantics):
    testing_labels = []
    adjacency_matrix = getAdjecencyMatrix(train_labels)
    for query_matrix_1xk in testing_latent_semantics:
        similarity_list = []        # it should be 1xn of float values.
        for each in training_latent_semantics:
            similarity_list.append(np.linalg.norm(np.array(each)-query_matrix_1xk))

        # It is a nob.
        count_seeds = 3
        new_similarity_list = {}

        for index, each in enumerate(similarity_list):
            new_similarity_list[str(index)] = each

        sorted_dict = sorted(new_similarity_list.items(), reverse=True, key=lambda kv: kv[1])

        seeds = []
        list_of_dict = list(sorted_dict)
        for i in range(count_seeds):
            seeds.append(list_of_dict[i][0])

        # TODO not considering 'n' as the limiting factor in no of relavant edges of an edge.
        # Initializing adjacency matrix.

        # TODO not sure if considering undirected edges will work for PPR.

        hubs_vs_authorities = get_hubs_authorities_from_adjacency_matrix(adjacency_matrix)
        transition_matrix = get_transition_matrix_from_hubs_authorities(hubs_vs_authorities)
        seed_nodes = get_seed_nodes(seeds, len(train_labels))
        ppr_matrix = get_page_ranking(0.4, transition_matrix, seed_nodes)
        # print(ppr_matrix)
        highest_type_ids = np.argsort(-ppr_matrix[:, -1])
        # print(highest_type_ids)
        # print('Most Relevant Type Id w.r.t to seed nodes')
        # This will return the index of the image. We have to pick the type of that image as output.
        print(highest_type_ids[:1])
        curr_label = train_labels[highest_type_ids[0]]
        testing_labels.append(curr_label)
        print('\n %s', {curr_label})
    return testing_labels