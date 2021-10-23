from dao_util import DAOUtil
from numpy.linalg import svd
from dimention_reduction_util import *
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
from k_means_util import reduce_dimensions_k_means
import numpy as np

#def write_to_file(data):

def similar_matrix(data, method='pearson'):
  data = np.array(data, dtype=np.float128)
  matrix = np.ones((data.shape[0],data.shape[0]))
  if method == 'euclidean':
    for i in range(0,data.shape[0]):
      for j in range(0,data.shape[0]):
        d = (sum((data[i]-data[j])*2))*0.5
        d=1/(1+d)
        matrix[i][j]=d
    return np.array(matrix)
  elif method == 'pearson':
    matrix = np.ones((data.shape[0],data.shape[0]))
    L=(data.T-data.mean(axis=1)).T
    for i in range(0,data.shape[0]):
      for j in range(0,data.shape[0]):
        d = np.sum(L[i]*L[j])/np.sqrt(np.sum(np.square(L[i]))*np.sum(np.square(L[j])))
        matrix[i][j]=d
    return np.array(matrix)

def get_image_label_array(feature_descriptors):
    image_label_array = []
    for feature_descriptor in feature_descriptors:
        label = feature_descriptor['label']
        image_type = label.split('-')[1]
        image_label_array.append(image_type)
        # image_label_array.append(i)
    return image_label_array


def find_reduced_type_similarity_matrix(dao_util, subject_id, feature_model, reduction_technique, k):
  images = dao_util.get_feature_descriptors_by_subject_id(str(subject_id))
  image_types = get_image_label_array(images)

  if(feature_model == 'color'):
    feature_extracted_dataset = np.zeros((len(images),192))
    for i in range(len(images)):
      feature_extracted_dataset[i] = np.array(images[i]['color_moment_feature_descriptor'])
  elif(feature_model == 'lbp'):
    feature_extracted_dataset = np.zeros((len(images),26))
    for i in range(len(images)):
      feature_extracted_dataset[i] = np.array(images[i]['elbp_feature_descriptor'])
  else:
    feature_extracted_dataset = np.zeros((len(images),1764))
    for i in range(len(images)):
      feature_extracted_dataset[i] = np.array(images[i]['hog_feature_descriptor'])

  # Go from 120xm to 12xm by averaging
  dict_type = {'cc': 0, 'con': 1, 'emboss': 2, 'jitter': 3, 'neg': 4,\
	'noise01': 5, 'noise02': 6, 'original': 7, 'poster': 8, 'rot': 9,\
	'smooth': 10, 'stipple': 11}
  type_weights = np.zeros((12, feature_extracted_dataset.shape[1]))
  for feature in range(feature_extracted_dataset.shape[1]):
    sum_type = np.zeros((12,2))
    for i in range(len(images)):
      sum_type[dict_type[image_types[i]]][0] += feature_extracted_dataset[i][feature]
      sum_type[dict_type[image_types[i]]][1] += 1
    for i in range(len(sum_type)):
      type_weights[i][feature] = sum_type[i][0]/sum_type[i][1]

  print("Calculating Similarity Matrix")
  similarity = similar_matrix(type_weights)

  if reduction_technique == 'pca':
    latent_type_features = get_reduced_matrix_using_pca(similarity, k)
  elif reduction_technique == 'svd':
    latent_type_features = get_reduced_matrix_using_svd(similarity, k)
  elif reduction_technique == 'lda':
    latent_type_features = get_reduced_matrix_using_lda(similarity, k)
  elif reduction_technique == 'kmeans':
    latent_type_features = reduce_dimensions_k_means(similarity, n_components=k, n_iterations=1000)

  return latent_type_features, similarity


##### Acquire parameters from user #####
feature = input("Enter feature model: ")
while not (feature in ["color", "lbp", "hog"]):
  print("Invalid input - choose from \"color\", \"lbp\", \"hog\"")
  feature = input("Enter feature model: ")

reduction_technique = input("Enter dimensionality reduction technique: ")
while not (reduction_technique in ["pca", "svd", "lda", "kmeans"]):
  print("Invalid input - choose from \"pca\", \"svd\", \"lda\", \"kmeans\"")
  reduction_technique = input("Enter dimensionality reduction technique: ")

k = int(input("Enter k value: "))


##### Compute Similarity Matrices #####
dao_util = DAOUtil()
Similarity_Matrix = []
Reduced_Matrix = []
for subject_id in range(1,41):
  print("Calculating on subject " + str(subject_id))
  reduced, sim = find_reduced_type_similarity_matrix(dao_util, subject_id, feature, reduction_technique, k)
  Similarity_Matrix.append(sim)
  Reduced_Matrix.append(reduced)



