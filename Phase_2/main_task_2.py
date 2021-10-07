from Util.dao_util import DAOUtil
from numpy.linalg import svd
from sklearn.decomposition import TruncatedSVD


def get_image_vector_matrix(feature_descriptors, feature_model):
    image_vector_matrix = []
    for feature_descriptor in feature_descriptors:
        image_vector_matrix.append(feature_descriptor[feature_model])
    return image_vector_matrix


def main():
    """
        Executes Task 2
    """
    """
        Connection to MongoDB using PyMongo
    """
    dao_util = DAOUtil()
    feature_model = input('Welcome to Task 2 Demo. Enter the feature model:')
    feature_model += '_feature_descriptor'
    subject_id = input('Enter Subject Id:')
    feature_descriptors = dao_util.get_feature_descriptors_by_subject_id(subject_id)
    image_vector_matrix = get_image_vector_matrix(feature_descriptors, feature_model)
    print(image_vector_matrix)
    # for row in image_vector_matrix:
    #     print(len(row))
    svd = TruncatedSVD(n_components=2)
    svd.fit_transform(image_vector_matrix)
    print(svd.singular_values_)
    # print(VT.shape)
    # print(VT)


main()
