import random
import numpy as np
import pandas as pd
import numpy as np


class KMeans:
    CENTROIDS = "centroids_kxm"
    INPUT_MATRIX = "matrix_nxm"
    OBJECTS_IN_K_DIMENSIONS = "matrix_nxk"
    DEFAULT_ITERATIONS = 100

    def serialize(self):
        return {KMeans.CENTROIDS: [lst.tolist() for lst in self.centroids],
                KMeans.INPUT_MATRIX: self.input_matrix.tolist(),
                KMeans.OBJECTS_IN_K_DIMENSIONS: self.objects_in_k_dimensions.tolist()}

    @staticmethod
    def deserialize(latent_feature_json_object):
        obj = KMeans(DEFAULT_ITERATIONS)
        obj.centroids = [np.array(lst) for lst in latent_feature_json_object[KMeans.CENTROIDS]]
        obj.input_matrix = latent_feature_json_object[KMeans.INPUT_MATRIX]
        obj.objects_in_k_dimensions = np.array(latent_feature_json_object[KMeans.OBJECTS_IN_K_DIMENSIONS])
        return obj

    def __init__(self, n_iterations) -> None:
        self.n_iterations = n_iterations

    def __init__(self):
        self.n_iterations = KMeans.DEFAULT_ITERATIONS

    @staticmethod
    def initialize_centroids(centroids, n_components, image_vector_matrix):
        random_number_set = set()
        for i in range(0, n_components):
            random_row_index = random.randint(0, len(image_vector_matrix) - 1)
            while random_row_index in random_number_set:
                random_row_index = random.randint(0, len(image_vector_matrix) - 1)
            random_number_set.add(random_row_index)
            # print(random_row_index)
            centroids.append(image_vector_matrix[random_row_index])

    @staticmethod
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

    @staticmethod
    def get_data_points(image_vector_matrix, point_vs_centroid, centroid_index):
        data_points = []
        for data_point_index in point_vs_centroid.keys():
            if point_vs_centroid[data_point_index] == centroid_index:
                data_points.append(image_vector_matrix[data_point_index])
        return np.array(data_points)

    @staticmethod
    def update_centroid_with_mean_of_cluster(centroids, point_vs_centroid, image_vector_matrix):
        for i in range(0, len(centroids)):
            data_points = KMeans.get_data_points(image_vector_matrix, point_vs_centroid, i)
            # print(data_points)
            mean = np.nanmean(data_points, axis=0)
            centroids[i] = mean

    @staticmethod
    def get_vectors_k_dimensions(image_vector_matrix, centroids):
        image_vector_matrix_k_dimensions = []
        for image_vector in image_vector_matrix:
            euclidean_distance_from_clusters = []
            for centroid in centroids:
                euclidean_distance = np.linalg.norm(image_vector - centroid)
                euclidean_distance_from_clusters.append(euclidean_distance)
            image_vector_matrix_k_dimensions.append(euclidean_distance_from_clusters)
        return image_vector_matrix_k_dimensions

    def compute(self, image_vector_matrix, n_components, *args):
        image_vector_matrix = np.array(image_vector_matrix)
        df = pd.DataFrame(image_vector_matrix)
        df.to_clipboard(index=False, header=False)
        centroids = []
        point_vs_centroid = {}
        KMeans.initialize_centroids(centroids, n_components, image_vector_matrix)
        # print(centroids)
        for i in range(0, self.n_iterations):
            KMeans.assign_points_to_centroids(point_vs_centroid, image_vector_matrix, centroids)
            KMeans.update_centroid_with_mean_of_cluster(centroids, point_vs_centroid, image_vector_matrix)
            # print(centroids)
            # print(centroid_vs_points)
        self.centroids = centroids
        self.input_matrix = image_vector_matrix
        image_vector_matrix_k_dimensions = KMeans.get_vectors_k_dimensions(image_vector_matrix, centroids)
        self.objects_in_k_dimensions = np.array(image_vector_matrix_k_dimensions).real
        return self.objects_in_k_dimensions

    def transform(self, image_vector_matrix):
       return np.array(KMeans.get_vectors_k_dimensions(image_vector_matrix,self.centroids))

