import numpy as np
from Constants import MEAN_WEIGHT
from Constants import STAND_DEV_WEIGHT
from Constants import SKEWNESS_WEIGHT
from Constants import COLOR_MOMENT_WEIGHT
from Constants import ELBP_WEIGHT
from Constants import HOG_WEIGHT
from Constants import NUMBER_OF_BINS

def get_similar_images_based_on_model(model_name, input_image_descriptor_object, db_descriptor_objects):
    """
    Computes the similar images based upon the input model
    :return List of similar images based upon the input model
    :rtype: List
    """

    """
        Compare the distance between images based upon input model. 'All' denotes combination of all models
    """
    label_vs_sorted_distances = None
    if model_name == 'color_moment':
        label_vs_sorted_distances = compare_color_moment(input_image_descriptor_object, db_descriptor_objects)
    elif model_name == 'elbp':
        label_vs_sorted_distances = compare_elbp_values(input_image_descriptor_object, db_descriptor_objects)
    elif model_name == 'hog':
        label_vs_sorted_distances = compare_hog_values(input_image_descriptor_object, db_descriptor_objects)
    elif model_name == 'all':
        label_vs_sorted_distances = compare_all_models(input_image_descriptor_object, db_descriptor_objects)
    output_list = list(label_vs_sorted_distances.items())
    output_list.sort(key=lambda x: x[1])
    return output_list


def get_euclidean_distance_between_color_moments(input_color_moment_feature_descriptor,
                                                 color_moment_feature_descriptor):
    """

    :param input_color_moment_feature_descriptor:
    :param color_moment_feature_descriptor:
    :return: Euclidean distance between color moments
    """
    input_color_moment_feature_descriptor = np.array(input_color_moment_feature_descriptor)
    color_moment_feature_descriptor = np.array(color_moment_feature_descriptor)

    # input_mean_vector = input_color_moment_feature_descriptor[:, :, 0]
    # given_mean_vector = color_moment_feature_descriptor[:, :, 0]

    # input_stand_dev_vector = input_color_moment_feature_descriptor[:, :, 1]
    # given_stand_dev_vector = color_moment_feature_descriptor[:, :, 1]
    #
    # input_skewness_vector = input_color_moment_feature_descriptor[:, :, 2]
    # given_skewness_vector = color_moment_feature_descriptor[:, :, 2]

    distance_between_means = np.linalg.norm(input_color_moment_feature_descriptor - color_moment_feature_descriptor)
    # distance_between_stand_dev = np.linalg.norm(input_stand_dev_vector - given_stand_dev_vector)
    # distance_between_skewness = np.linalg.norm(input_skewness_vector - given_skewness_vector)

    # total_distance = MEAN_WEIGHT * distance_between_means + STAND_DEV_WEIGHT * distance_between_stand_dev + SKEWNESS_WEIGHT * distance_between_skewness
    return distance_between_means


def compare_color_moment(input_image_descriptor_object, db_descriptor_objects):
    """
    :param input_image_descriptor_object:
    :param db_descriptor_objects:
    :return: Compares color moment feature between input image and other images.
    Returns euclidean distance between the images
    """
    label_vs_euclidean_distance = {}
    input_color_moment_feature_descriptor = input_image_descriptor_object['color_moment_feature_descriptor']
    for db_descriptor_object in db_descriptor_objects:
        image_label = db_descriptor_object['label']
        color_moment_feature_descriptor = db_descriptor_object['color_moment_feature_descriptor']
        euclidean_distance = get_euclidean_distance_between_color_moments(input_color_moment_feature_descriptor,
                                                                          color_moment_feature_descriptor)
        label_vs_euclidean_distance[image_label] = euclidean_distance
    return label_vs_euclidean_distance


def get_elbp_histogram(elbp_feature_descriptor):
    """
    :param elbp_feature_descriptor:
    :return: Histogram of the ELBP feature descriptor
    """
    num_points = 24
    (hist, _) = np.histogram(elbp_feature_descriptor.ravel(), bins=range(0, num_points + 3),
                             range=(0, num_points + 2))

    eps = 1e-7
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)
    return hist


def compare_elbp_values(input_image_descriptor_object, db_descriptor_objects):
    """
    :param input_image_descriptor_object:
    :param db_descriptor_objects:
    :return: compares elbp features of the input image with other images.
    Returns chi square distance between the images.
    """
    label_vs_euclidean_distance = {}
    input_elbp_feature_descriptor = np.array(input_image_descriptor_object['elbp_feature_descriptor'])
    # input_histogram_distance = get_elbp_histogram(input_elbp_feature_descriptor)
    for db_descriptor_object in db_descriptor_objects:
        image_label = db_descriptor_object['label']
        elbp_feature_descriptor = np.array(db_descriptor_object['elbp_feature_descriptor'])
        # histogram_distance = get_elbp_histogram(elbp_feature_descriptor)
        dist = 0.5 * np.sum(((input_elbp_feature_descriptor - elbp_feature_descriptor) ** 2) /
                            (input_elbp_feature_descriptor + elbp_feature_descriptor + 1e-10))
        label_vs_euclidean_distance[image_label] = dist
    return label_vs_euclidean_distance


def compare_hog_values(input_image_descriptor_object, db_descriptor_objects):
    """
    :param input_image_descriptor_object:
    :param db_descriptor_objects:
    :return: compares hog features of the input image with other images.
    Returns earth movers distance between the images.
    """
    label_vs_emd_distance = {}
    input_hog_feature_descriptor = np.array(input_image_descriptor_object['hog_feature_descriptor'])
    for db_descriptor_object in db_descriptor_objects:
        image_label = db_descriptor_object['label']
        hog_feature_descriptor = np.array(db_descriptor_object['hog_feature_descriptor'])
        # for i in range(0, NUMBER_OF_BINS):
        input_bin_vector = input_hog_feature_descriptor
        compare_bin_vector = hog_feature_descriptor
        eucledian_distance_between_bins = np.linalg.norm(input_bin_vector - compare_bin_vector)
        emd_distance = eucledian_distance_between_bins
        label_vs_emd_distance[image_label] = emd_distance
    return label_vs_emd_distance


def normalize(res):
    """
    :param res:
    :return: Normalizes the distance measure.
    Formula: (value-min)/(max-min)
    """
    list_res = list(res.items())
    list_res.sort(key=lambda x: x[1])
    length = len(list_res)
    # print(list_res)
    min_element = list_res[0][1]
    max_element = list_res[length - 1][1]
    for i in range(0, length):
        item = list(list_res[i])
        item[1] = (item[1] - min_element) / (max_element - min_element)
        list_res[i] = item
    print(dict(list_res))
    return dict(list_res)


def combine_all_models(normalized_color_moment_res, normalized_elbp_res, normalized_hog_res):
    """
    :param normalized_color_moment_res:
    :param normalized_elbp_res:
    :param normalized_hog_res:
    :return: Combines all the models as per the weighted mean
    """
    final_res = {}
    for key in normalized_color_moment_res.keys():
        final_res[key] = COLOR_MOMENT_WEIGHT * normalized_color_moment_res[key] + ELBP_WEIGHT * normalized_elbp_res[
            key] + HOG_WEIGHT * normalized_hog_res[key]
    return final_res


def compare_all_models(input_image_descriptor_object, db_descriptor_objects):
    """
    :param input_image_descriptor_object:
    :param db_descriptor_objects:
    :return: Computes all model distances, normalizes it and combines it using weighted average
    """
    color_moment_res = compare_color_moment(input_image_descriptor_object, db_descriptor_objects)
    elbp_res = compare_elbp_values(input_image_descriptor_object, db_descriptor_objects)
    hog_res = compare_hog_values(input_image_descriptor_object, db_descriptor_objects)
    normalized_color_moment_res = normalize(color_moment_res)
    normalized_elbp_res = normalize(elbp_res)
    normalized_hog_res = normalize(hog_res)
    res = combine_all_models(normalized_color_moment_res, normalized_elbp_res, normalized_hog_res)
    return res
