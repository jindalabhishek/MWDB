import numpy

from Util.dao_util import DAOUtil
from numpy.linalg import svd
from dimention_reduction_util import *


def get_image_vector_matrix(feature_descriptors, feature_model):
    image_vector_matrix = []
    image_types = []
    for feature_descriptor in feature_descriptors:
        image_vector_matrix.append(feature_descriptor[feature_model])
        image_types.append(feature_descriptor['label'].split("-")[2])
    return image_vector_matrix, image_types


def task1():
    """
        Executes Task 1
        Output Subject - latent semantics matrix, (subject-list of weight matrix)
    """
    """
        Connection to MongoDB using PyMongo
    """
    
    ##### Acquire parameters from user #####
    feature = input("Enter feature model: ")
    while not (feature in ["color", "lbp", "hog"]):
      print("Invalid input - choose from \"color\", \"lbp\", \"hog\"")
      feature = input("Enter feature model: ")

    type_list = ['cc', 'con', 'emboss', 'jitter', 'neg',\
	  'noise01', 'noise02', 'original', 'poster', 'rot',\
	  'smooth', 'stipple']
    img_type = input("Enter image type: ")
    while not (img_type in type_list):
      print("Invalid input - choose from \"cc\", \"con\", \"detail\", \"emboss\", \"jitter\", \"neg\", \"noise1\", \"noise2\", \"original\", \"poster\", \"rot\", \"smooth\", \"stipple\"")
      img_type = input("Enter image type: ")

    k = int(input("Enter k value: "))
    
    reduction_technique = input("Enter dimensionality reduction technique: ")
    while not (reduction_technique in ["pca", "svd", "lda", "kmeans"]):
      print("Invalid input - choose from \"pca\", \"svd\", \"lda\", \"kmeans\"")
      reduction_technique = input("Enter dimensionality reduction technique: ")

    ##### Extract and preprocess data matrix #####
    db = DAOUtil()
    images = db.get_feature_descriptors_by_type_id(img_type)

    if(feature == 'color'):
      feature_extracted_dataset = np.zeros((len(images),192))
      for i in range(len(images)):
        feature_extracted_dataset[i] = np.array(images[i]['color_moment_feature_descriptor'])
    elif(feature == 'lbp'):
      feature_extracted_dataset = np.zeros((len(images),26))
      for i in range(len(images)):
        feature_extracted_dataset[i] = np.array(images[i]['elbp_feature_descriptor'])
    else:
      feature_extracted_dataset = np.zeros((len(images),1764))
      for i in range(len(images)):
        feature_extracted_dataset[i] = np.array(images[i]['hog_feature_descriptor'])

    if reduction_technique == 'pca':
        subject_weight_matrix = get_reduced_matrix_using_pca(feature_extracted_dataset, k)
    elif reduction_technique == 'svd':
        subject_weight_matrix = get_reduced_matrix_using_svd(feature_extracted_dataset, k)
    elif reduction_technique == 'lda':
        subject_weight_matrix = get_reduced_matrix_using_lda(feature_extracted_dataset, k)
    elif reduction_technique == 'kmeans':
        subject_weight_matrix = get_reduced_matrix_using_kmeans(feature_extracted_dataset, k)
        
    subject_weight_pairs = np.zeros((40,k))

    for latent_feature in range(k):
      sum_subject = np.zeros((40,2))
      for i in range(len(images)):
        subject_id = int(images[i]['label'].split('-',3)[2])-1
        sum_subject[subject_id][0] += subject_weight_matrix[i][latent_feature]
        sum_subject[subject_id][1] += 1
      for i in range(40):
        subject_weight_pairs[i][latent_feature] = sum_subject[i][0]/sum_subject[i][1]
    return subject_weight_pairs

subject_weight_pairs = task1()
for i in range(len(subject_weight_pairs)):
  print("Subject " + str(i+1) + ":", end=' ')
  for j in range(len(subject_weight_pairs[i])):
    print(str(subject_weight_pairs[i][j]), end=', ')
