from math import sqrt


def get_mean(image):
    """
    :param image:
    :return: Mean of image pixels
    """
    res = 0
    total_length = len(image) * len(image[0])
    for image_pixels in image:
        for pixel in image_pixels:
            res += pixel
    mean = res / total_length
    return mean


def get_standard_deviation(images, mean):
    """
    :param images:
    :param mean:
    :return: Standard Deviation of image pixels
    """
    res = 0
    total_length = len(images) * len(images[0])
    for image_pixels in images:
        for pixel in image_pixels:
            temp = pixel-mean
            res += pow(temp, 2)
    res = res/total_length
    standard_deviation = sqrt(res)
    return standard_deviation


def get_skewness(image, mean, standard_deviation):
    """
    :param image:
    :param mean:
    :param standard_deviation:
    :return: Skewness of image_pixels
    """
    res = 0.0
    total_length = len(image) * len(image[0])
    for image_pixels in image:
        for pixel in image_pixels:
            temp = pixel - mean
            res += pow(temp, 3)
    cube_stand_dev = pow(standard_deviation, 3)
    skewness = res / ((total_length-1) * cube_stand_dev)
    return skewness
