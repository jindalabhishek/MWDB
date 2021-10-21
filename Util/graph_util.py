import numpy as np
from numpy.linalg import svd


def convert_similarity_matrix_to_graph(similarity_matrix, n, m):
    adjacency_matrix = np.zeros((len(similarity_matrix), len(similarity_matrix)))
    for i in range(0, len(similarity_matrix)):
        sorted_indexes = np.argsort(-similarity_matrix[i])
        print(sorted_indexes)
        for j in range(0, min(n, len(sorted_indexes))):
            adjacency_matrix[i][sorted_indexes[j]] = 1
    return adjacency_matrix


def get_hubs_authorities_from_adjacency_matrix(adjacency_matrix):
    return np.transpose(adjacency_matrix)


def get_transition_matrix_from_hubs_authorities(hubs_vs_authorities):
    column_sum = hubs_vs_authorities.sum(axis=0)
    print(column_sum)
    transition_matrix = hubs_vs_authorities / column_sum
    return transition_matrix


def get_seed_nodes(subject_ids_set, k):
    seed_nodes = np.zeros((k, 1))
    n_subject_ids = len(subject_ids_set)
    probability = 1 / (n_subject_ids + 1)
    remaining_subjects = k - n_subject_ids
    remaining_subjects_probability = probability * (1 / remaining_subjects)
    for i in range(0, k):
        if i + 1 in subject_ids_set:
            seed_nodes[i][0] = probability
        else:
            seed_nodes[i][0] = remaining_subjects_probability
    return seed_nodes


def get_page_ranking(beta, transition_matrix, seed_nodes):
    identity_matrix = np.identity(len(transition_matrix))
    print(identity_matrix)
    transition_matrix = (1 - beta) * transition_matrix
    matrix = np.subtract(identity_matrix, transition_matrix)
    inverse_matrix = np.linalg.inv(matrix)
    seed_nodes = beta * seed_nodes
    page_ranking = np.matmul(inverse_matrix, seed_nodes)
    return page_ranking
