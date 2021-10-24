# Assuming the image is provided in the form of 64x64 and input the same feature model as used in the input latent semantic.
# Assuming I have all the required data in the json file stored.

import os
import json
import numpy as np

from Phase_1.vector_util import *
from Phase_1.feature_descriptor_util import *
from Phase_1.image_comparison_util import *
from Constants import GREY_SCALE_MAX
from Phase_1.image_comparison_util import get_similar_images_based_on_model
from Util.dao_util import *
from task567_util import *
# from Util.json_util import LatentSemanticFile
# from Util.json_util import LatentSemanticFile
image_path = '/Users/dhruv/Desktop/Sem1/MWDB/project2/all/image-neg-1-2.png'
    # input('Welcome to Task 5 Demo. Enter Full Path of the image for query: ')
print('Select your feature descriptor! (color_moment, elbp, hog): ')
feature_model = input()
# feature_model += 'feature_descriptor'
# /Users/dhruv/Desktop/Sem1/MWDB/project2/query/cc-image-2.png

daoUtil = DAOUtil()

output1_fd = open('../Outputs/task_1_hog_LDA_cc.json')
all_latent_semantics = json.load(output1_fd)

# lsf = LatentSemanticFile()

query_matrix_1xm = []

"""
   Compute the image pixels for the image
"""
image_pixels = convert_image_to_matrix(image_path)
print('Image Size:', len(image_pixels), len(image_pixels[0]), 'Max Pixel Size:', np.amax(image_pixels))
"""
   Normalize the image pixels
"""
image_pixels = image_pixels / GREY_SCALE_MAX
"""
    Compute All the feature descriptors
"""

# TODO Assuming these descriptors are 1xm of query image
if feature_model == 'color_moment':
    color_moment_feature_descriptor = get_color_moment_feature_descriptor(image_pixels)
    color_moment_feature_descriptor = get_reshaped_color_moment_vector(color_moment_feature_descriptor)
    query_matrix_1xm = color_moment_feature_descriptor.copy()
elif feature_model == 'elbp':
    elbp_feature_descriptor = get_elbp_feature_descriptor(image_pixels)
    elbp_feature_descriptor = get_elbp_histogram(elbp_feature_descriptor)
    query_matrix_1xm = elbp_feature_descriptor.copy()
elif feature_model == 'hog':
    hog_feature_descriptor = get_hog_feature_descriptor(image_pixels)
    query_matrix_1xm = hog_feature_descriptor.copy()

# Fetching all data.
all_data = daoUtil.get_feature_descriptors_for_all_images()

# Extracting nxm
# all_data_matrix_nxm = [each[str(feature_model) + '_feature_descriptor'] for each in all_data]
# all_labels =[each['label'] for each in all_data]
feature_model_name = feature_model + '_feature_descriptor'
# Converted all data nxm to nxk
reduced_all_data_matrix_nxk = get_reduced_dimension_nxk_using_latent_semantics(all_data, all_latent_semantics, feature_model_name)

# Converted query 1xm to 1xk
reduced_query_1xk = {feature_model_name: transform_1xm_to_1xk(query_matrix_1xm, all_latent_semantics)}

# labelled_reduced_all_data_matrix_nxk
output_list = get_similar_images_based_on_model(feature_model, reduced_query_1xk, reduced_all_data_matrix_nxk)
print(output_list)

n_value = int(input('Enter n, for top n similar images'))

output_list_n = output_list[:n_value]

print(output_list_n)