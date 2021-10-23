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

def find_reduced_subject_similarity_matrix(dao_util, img_type, feature_model, reduction_technique, k):
  images = dao_util.get_feature_descriptors_by_type_id(img_type)

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

  # Go from 400xm to 40xm by averaging
  subject_weights = np.zeros((40, feature_extracted_dataset.shape[1]))
  for feature in range(feature_extracted_dataset.shape[1]):
    sum_subject = np.zeros((40,2))
    for i in range(len(images)):
      subject_id = int(images[i]['label'].split('-',3)[2])-1
      sum_subject[subject_id][0] += feature_extracted_dataset[i][feature]
      sum_subject[subject_id][1] += 1
    for i in range(40):
      subject_weights[i][feature] = sum_subject[i][0]/sum_subject[i][1]

  print("Calculating Similarity Matrix")
  similarity = similar_matrix(subject_weights)

  if reduction_technique == 'pca':
    latent_subject_features = get_reduced_matrix_using_pca(similarity, k)
  elif reduction_technique == 'svd':
    latent_subject_features = get_reduced_matrix_using_svd(similarity, k)
  elif reduction_technique == 'lda':
    latent_subject_features = get_reduced_matrix_using_lda(similarity, k)
  elif reduction_technique == 'kmeans':
    latent_subject_features = reduce_dimensions_k_means(similarity, n_components=k, n_iterations=1000)

  return latent_subject_features, similarity


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
img_types = ['cc', 'con', 'emboss', 'jitter', 'neg',\
	'noise01', 'noise02', 'original', 'poster', 'rot',\
	'smooth', 'stipple']
dao_util = DAOUtil()
Similarity_Matrix = []
Reduced_Matrix = []
for type in img_types:
  print("Calculating on type " + type)
  reduced, sim = find_reduced_subject_similarity_matrix(dao_util, type, feature, reduction_technique, k)
  Similarity_Matrix.append(sim)
  Reduced_Matrix.append(reduced)



