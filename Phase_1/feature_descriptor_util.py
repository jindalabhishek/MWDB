import numpy as np
import color_moment
from skimage.feature import hog
from skimage.feature import local_binary_pattern
from Constants import MEAN_WEIGHT
from Constants import STAND_DEV_WEIGHT
from Constants import SKEWNESS_WEIGHT


def get_reshaped_color_moment_vector(color_moment_feature_descriptor):
    return color_moment_feature_descriptor.reshape(1, -1)


def get_color_moment_feature_descriptor(image_pixels):
    """
    :param image_pixels:
    :return: Color Moment feature descriptor
    """
    window_size = 8
    image_size = 64
    color_moment_params = 3

    color_moment_feature_descriptor = np.zeros((window_size, window_size, color_moment_params))
    face_image = np.array(image_pixels)
    a = 0
    for j in range(0, image_size, window_size):
        b = 0
        for k in range(0, image_size, window_size):
            """
            Breaking image into block of 8x8
            """
            image_block = face_image[j:j + window_size, k:k + window_size]
            local_mean = color_moment.get_mean(image_block)
            local_stand_dev = color_moment.get_standard_deviation(image_block, local_mean)
            local_skew = color_moment.get_skewness(image_block, local_mean, local_stand_dev)
            color_moment_feature_descriptor[a][b][0] = local_mean
            color_moment_feature_descriptor[a][b][1] = local_stand_dev
            color_moment_feature_descriptor[a][b][2] = local_skew
            b += 1
        a += 1
    return color_moment_feature_descriptor


def get_elbp_feature_descriptor(image_pixels):
    """
    :param image_pixels:
    :return: ELBP feature descriptor
    """
    return local_binary_pattern(image_pixels, 24, 8, method="uniform")


def get_hog_feature_descriptor(image_pixels):
    """
    :param image_pixels:
    :return: HOG Feature Descriptor
    """
    hog_feature_descriptor, hog_image = hog(image_pixels, orientations=9,
                                            pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True,
                                            feature_vector=True)
    return hog_feature_descriptor
