import json

from Util.dao_util import DAOUtil
from sklearn.decomposition import LatentDirichletAllocation
from Util.k_means_util import reduce_dimensions_k_means
from Util.graph_util import *
import numpy as np
from Util.json_util import LatentSemanticFile
from dimensionality_reduction import LDA
from Util.Utils import *

def get_image_vector_matrix(feature_descriptors, feature_model):
    image_vector_matrix = []
    for feature_descriptor in feature_descriptors:
        image_vector_matrix.append(feature_descriptor[feature_model])
    return image_vector_matrix


def get_image_label_array(feature_descriptors):
    image_label_array = []
    for feature_descriptor in feature_descriptors:
        label = feature_descriptor['label']
        image_type = label.split('-')[1]
        image_label_array.append(image_type)
        # image_label_array.append(i)
    return image_label_array


def calculate_lda(image_vector_matrix, image_types, components):
    lda = LatentDirichletAllocation(n_components=components)
    image_vector_matrix_lda = lda.fit_transform(image_vector_matrix, image_types)
    return image_vector_matrix_lda


def normalize_data_for_lda(image_vector_matrix):
    normalized_data = (image_vector_matrix - np.min(image_vector_matrix)) \
                      / (np.max(image_vector_matrix) - np.min(image_vector_matrix))
    return normalized_data


def main():
    """
        Executes Task 9
        Output Subject - latent semantics matrix, (subject-list of weight matrix)
    """
    """
        Connection to MongoDB using PyMongo
    """
    input_path = input('Welcome to Task 9 Demo. Enter the file which contains similarity matrix:')
    similarity_matrix = get_similarity_matrix(input_path)
    similarity_matrix = np.array(similarity_matrix)
    n = int(input('Enter value of n:'))
    m = int(input('Enter value of m:'))
    subject_id_1 = int(input('Subject Id1:'))
    subject_id_2 = int(input('Subject Id2:'))
    subject_id_3 = int(input('Subject Id3:'))
    subject_ids = set()
    subject_ids.add(subject_id_1)
    subject_ids.add(subject_id_2)
    subject_ids.add(subject_id_3)

    adjacency_matrix = convert_similarity_matrix_to_graph(similarity_matrix, n)
    hubs_vs_authorities = get_hubs_authorities_from_adjacency_matrix(adjacency_matrix)
    transition_matrix = get_transition_matrix_from_hubs_authorities(hubs_vs_authorities)
    seed_nodes = get_seed_nodes(subject_ids, len(similarity_matrix))
    ppr_matrix = get_page_ranking(0.4, transition_matrix, seed_nodes)
    # print(ppr_matrix)
    highest_subject_ids = np.argsort(-ppr_matrix[:, -1])
    print(highest_subject_ids)
    print('Most Relevant M Subjects Ids w.r.t to seed nodes')
    print(highest_subject_ids[:m]+1)


main()
