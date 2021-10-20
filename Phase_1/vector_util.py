import cv2


def convert_image_to_matrix(image_path):
    """
    :param image_path:
    :return: array of image pixels of size 64x64
    """
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)