import os
import feature_descriptor_util
import image_comparison_util
import vector_util
from Util.dao_util import DAOUtil
import numpy as np

grey_scale_max = 256


def main():
    """
        Executes Task 2
    """
    """
        Connection to MongoDB using PyMongo
    """
    dao_util = DAOUtil()

    folder_path = input('Welcome to Task 2 Demo. Enter Full Path of the folder containing the images:')
    """
        Clear the MongoDB for fresh run
    """
    dao_util.delete_records()
    """
        Compute Base Path from full path such as set1, set2 or set3
    """
    parent_dir = folder_path.split('\\')
    parent_dir = parent_dir[len(parent_dir) - 1]
    print('Base Path:', parent_dir)
    for name in os.listdir(folder_path):
        """
           Compute the image pixels for the image
        """
        image_pixels = vector_util.convert_image_to_matrix(folder_path + '\\' + name)
        print('Image Size:', len(image_pixels), len(image_pixels[0]), 'Max Pixel Size:', np.amax(image_pixels))
        """
           Normalize the image pixels
        """
        image_pixels = image_pixels / grey_scale_max
        """
            Compute All the feature descriptors
        """
        color_moment_feature_descriptor = feature_descriptor_util.get_color_moment_feature_descriptor(image_pixels)
        color_moment_feature_descriptor = feature_descriptor_util\
            .get_reshaped_color_moment_vector(color_moment_feature_descriptor)

        elbp_feature_descriptor = feature_descriptor_util.get_elbp_feature_descriptor(image_pixels)
        elbp_feature_descriptor = image_comparison_util.get_elbp_histogram(elbp_feature_descriptor)
        hog_feature_descriptor = feature_descriptor_util.get_hog_feature_descriptor(image_pixels)

        """
            Image Feature Descriptor object to be saved in DB. 
            Note: Converts each descriptor from numpy array to list, otherwise MongoDb treats it
            as a foreign object and throws an error
        """
        feature_descriptor = {'remote_base_path': parent_dir,
                              'label': name,
                              'color_moment_feature_descriptor': color_moment_feature_descriptor,
                              'elbp_feature_descriptor': elbp_feature_descriptor.tolist(),
                              'hog_feature_descriptor': hog_feature_descriptor.tolist()}
        """
            Save the Image Feature Descriptor object to DB. Please note that ideally, we should insert all records
            in one go. I am inserting record one by one for demo purpose only.
        """
        db_feature_descriptor_id = dao_util.save_to_db(feature_descriptor)
        print(db_feature_descriptor_id)


main()
