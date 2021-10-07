from PIL import Image
import numpy as np


def convert_image_to_matrix(image_path):
    """
    :param image_path:
    :return: array of image pixels of size 64x64
    """
    image = Image.open(image_path, 'r')
    return np.array(image)
