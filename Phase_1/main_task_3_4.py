from Util.dao_util import DAOUtil
from image_comparison_util import compare_all_models
from image_comparison_util import compare_color_moment
from image_comparison_util import compare_elbp_values
from image_comparison_util import compare_hog_values
from model import ALL
from model import COLOR_MOMENT
from model import ELBP
from model import HOG


def get_index_of_input_object(image_label, db_descriptor_objects):
    """
        Computes the index of input image in DB objects
        :return Index of input image in DB objects
        :rtype: Integer
    """
    res_index = 0
    for i in range(0, len(db_descriptor_objects)):
        """
            Check if the label matches the input label.
        """
        if db_descriptor_objects[i]['label'] == image_label:
            res_index = i
            break
    return res_index


def get_similar_images_based_on_model(model_name, input_image_descriptor_object, db_descriptor_objects):
    """
    Computes the similar images based upon the input model
    :return List of similar images based upon the input model
    :rtype: List
    """

    """
        Compare the distance between images based upon input model. 'All' denotes combination of all models
    """
    if model_name == COLOR_MOMENT:
        label_vs_sorted_distances = compare_color_moment(input_image_descriptor_object, db_descriptor_objects)
    elif model_name == ELBP:
        label_vs_sorted_distances = compare_elbp_values(input_image_descriptor_object, db_descriptor_objects)
    elif model_name == HOG:
        label_vs_sorted_distances = compare_hog_values(input_image_descriptor_object, db_descriptor_objects)
    elif model_name == ALL:
        label_vs_sorted_distances = compare_all_models(input_image_descriptor_object, db_descriptor_objects)
    output_list = list(label_vs_sorted_distances.items())
    output_list.sort(key=lambda x: x[1])
    return output_list


def main():
    """
        Executes Task 3 & 4. Please execute Task-2 before running Task 3 & 4.
    """
    """
        Connection to MongoDB using PyMongo
    """
    dao_util = DAOUtil()
    remote_base_path = str(input('Enter Base Path:'))
    image_label = str(input('Enter Image Label:'))
    model_name = str(input('Enter the model name out of (cm8x8, elbp, hog, all):'))
    k = int(input('Enter the value of k to find k similar images in base path:'))
    db_query_object = {'remote_base_path': remote_base_path}
    """
        Loads all the images contained in remote base path
    """
    db_descriptor_objects = list(dao_util.get_records(db_query_object))
    """
        Computes the index of input image.
    """
    index_of_input_image = get_index_of_input_object(image_label, db_descriptor_objects)
    input_image_descriptor_object = db_descriptor_objects[index_of_input_image]
    """
        Delete the input image from descriptor objects to be compared.
    """
    del db_descriptor_objects[index_of_input_image]
    """
        Get K similar images based on model.
    """
    sorted_distance_list = get_similar_images_based_on_model(model_name,
                                                             input_image_descriptor_object, db_descriptor_objects)
    """
        Slice the output list to k images.
    """
    print('K Similar Images', sorted_distance_list[:k])


main()
