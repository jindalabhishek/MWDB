# using pca, k = 5 by default.

def create_similarity_matrix_from_images(folder_1_path):
    latent_semantics = get_latent_semantic_file(folder_1_path)
