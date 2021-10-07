from sklearn.decomposition import TruncatedSVD, PCA


def get_reduced_matrix_using_pca(image_vector_matrix, n_components):
    pca = PCA(n_components)
    return pca.fit_transform(image_vector_matrix)


def get_reduced_matrix_using_svd(image_vector_matrix, n_components):
    svd = TruncatedSVD(n_components)
    VT = svd.fit_transform(image_vector_matrix)
    print(svd.singular_values_)
    print(VT.shape)
    print(VT)