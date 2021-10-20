import numpy as np


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
