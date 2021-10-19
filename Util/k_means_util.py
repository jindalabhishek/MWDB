import random
import pandas as pd
import numpy as np


def initialize_centroids(centroids, n_components, image_vector_matrix):
    random_number_set = set()
    for i in range(0, n_components):
        random_row_index = random.randint(0, len(image_vector_matrix) - 1)
        while random_row_index in random_number_set:
            random_row_index = random.randint(0, len(image_vector_matrix) - 1)
        random_number_set.add(random_row_index)
        # print(random_row_index)
        centroids.append(image_vector_matrix[random_row_index])


def assign_points_to_centroids(point_vs_centroid, image_vector_matrix, centroids):
    for i in range(0, len(image_vector_matrix)):
        min_euclidean_distance = float('inf')
        min_centroid = -1
        for j in range(0, len(centroids)):
            euclidean_distance = np.linalg.norm(image_vector_matrix[i] - centroids[j])
            if euclidean_distance < min_euclidean_distance:
                min_centroid = j
                min_euclidean_distance = euclidean_distance
        point_vs_centroid[i] = min_centroid


def get_data_points(image_vector_matrix, point_vs_centroid, centroid_index):
    data_points = []
    for data_point_index in point_vs_centroid.keys():
        if point_vs_centroid[data_point_index] == centroid_index:
            data_points.append(image_vector_matrix[data_point_index])
    return np.array(data_points)


def update_centroid_with_mean_of_cluster(centroids, point_vs_centroid, image_vector_matrix):
    for i in range(0, len(centroids)):
        data_points = get_data_points(image_vector_matrix, point_vs_centroid, i)
        # print(data_points)
        mean = np.nanmean(data_points, axis=0)
        centroids[i] = mean


def get_vectors_k_dimensions(image_vector_matrix, centroids):
    image_vector_matrix_k_dimensions = []
    for image_vector in image_vector_matrix:
        euclidean_distance_from_clusters = []
        for centroid in centroids:
            euclidean_distance = np.linalg.norm(image_vector - centroid)
            euclidean_distance_from_clusters.append(euclidean_distance)
        image_vector_matrix_k_dimensions.append(euclidean_distance_from_clusters)
    return image_vector_matrix_k_dimensions


def reduce_dimensions_k_means(image_vector_matrix, n_components, n_iterations):
    image_vector_matrix = np.array(image_vector_matrix)
    df = pd.DataFrame(image_vector_matrix)
    df.to_clipboard(index=False, header=False)
    centroids = []
    point_vs_centroid = {}
    initialize_centroids(centroids, n_components, image_vector_matrix)
    # print(centroids)
    for i in range(0, n_iterations):
        assign_points_to_centroids(point_vs_centroid, image_vector_matrix, centroids)
        update_centroid_with_mean_of_cluster(centroids, point_vs_centroid, image_vector_matrix)
        # print(centroids)
        # print(centroid_vs_points)
    image_vector_matrix_k_dimensions = get_vectors_k_dimensions(image_vector_matrix, centroids)
    return image_vector_matrix_k_dimensions
    # print(point_vs_centroid)
    centroid_vs_points = {}
    # for data_point in point_vs_centroid.keys():
    #     centroid = point_vs_centroid[data_point]
    #     if centroid_vs_points.get(centroid):
    #         centroid_vs_points[centroid].append(data_point)
    #     else:
    #         centroid_vs_points[centroid] = [data_point]
    # print(centroid_vs_points)
    # print(image_vector_matrix_k_dimensions)
    # print(len(image_vector_matrix_k_dimensions))
