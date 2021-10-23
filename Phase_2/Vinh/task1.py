import numpy as np
from Util.dao_util import DAOUtil
from Dimensionality_Reduction import dimensionality_reduction

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
  print("Invalid input - choose from \"cc\", \"con\", \"detail\", \
\"emboss\", \"jitter\", \"neg\", \"noise1\", \"noise2\", \"original\", \
\"poster\", \"rot\", \"smooth\", \"stipple\"")
  img_type = input("Enter image type: ")

k = int(input("Enter k value: "))

reduction_technique = input("Enter dimensionality reduction technique: ")
while not (reduction_technique in ["pca", "svd", "lda", "kmean"]):
  print("Invalid input - choose from \"pca\", \"svd\", \"lda\", \"kmean\"")
  reduction_technique = input("Enter dimensionality reduction technique: ")

##### Extract and preprocess data matrix #####
db = DAOUtil()
images = db.get_feature_descriptors_by_type_id(img_type)

if(feature == 'color'):
  feature_extracted_dataset = np.zeros((len(images),192))  # TO BE FIXED
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

latent_features_dataset = dimensionality_reduction(feature_extracted_dataset, k, reduction_technique)

# Going from 400xk to 40xk by averaging 10 samples/subject
subject_weight_pairs = np.zeros((40,k))
for latent_feature in range(k):
  sum_subject = np.zeros((40,2))
  for i in range(len(images)):
    subject_id = int(images[i]['label'].split('-',3)[2])-1
    sum_subject[subject_id][0] += latent_features_dataset[i][latent_feature]
    sum_subject[subject_id][1] += 1
  for i in range(40):
    subject_weight_pairs[i][latent_feature] = sum_subject[i][0]/sum_subject[i][1]

# -> OUTPUT = subject_weight_pairs (40 x k)
