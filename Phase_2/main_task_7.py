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
import re

image_path = input('Enter path to query image: ')
# input('Welcome to Task 5 Demo. Enter Full Path of the image for query: ')
# '/Users/dhruv/Desktop/dhruv_gray.jpeg'

input_file = input('Enter path to latent semantic file: ')


task_number = input_file.split('/')[-1][:6]

feature_model = input('Select your feature descriptor! (color_moment, elbp, hog), if task 1 or 2 input file, input correspondingly: ')
# feature_model += 'feature_descriptor'
# /Users/dhruv/Desktop/Sem1/MWDB/project2/query/cc-image-2.png

daoUtil = DAOUtil()

output1_fd = open(input_file)
all_latent_semantics = json.load(output1_fd)

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

# # Extracting nxm
# all_data_matrix_nxm = [each[str(feature_model) + '_feature_descriptor'] for each in all_data]

feature_model_name = feature_model + '_feature_descriptor'

# Converted all data nxm to nxk
reduced_all_data_matrix_nxk,matrices = get_reduced_dimension_nxk_using_latent_semantics(all_data, all_latent_semantics, feature_model_name, task_number)

# Converted query 1xm to 1xk
reduced_query_1xk = {feature_model_name: transform_1xm_to_1xk(query_matrix_1xm, all_latent_semantics, task_number)}

output_list = get_similar_images_based_on_model(feature_model, reduced_query_1xk, reduced_all_data_matrix_nxk)
# print(output_list)

truncated_output_list = output_list[:10]
print(truncated_output_list)
dict_of_types = {}
for i in range(10):
    curr_label = truncated_output_list[i][0]
    # regex = re.compile(r'image-\w*-\d-\d.png', re.IGNORECASE)
    # matching = regex.search(curr_label)
    # curr_type = matching.group(1)                   # assuming group : [type, subject, sample] 0 indexed.

    split_curr_label = curr_label.split('-')
    curr_type = split_curr_label[2]
    # store matching in dict
    if dict_of_types.get(curr_type) is None:
        dict_of_types[curr_type] = (10-i)
    else:
        dict_of_types[curr_type] += (10-i)

query_subject = max(zip(dict_of_types.values(), dict_of_types.keys()))[1]
print(query_subject)











