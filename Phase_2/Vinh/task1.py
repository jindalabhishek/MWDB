import numpy as np
import readData
from FeatureExtractions import extract_feature
from Dimensionality_Reduction import dimensionality_reduction

database = readData.read_data()

##### Acquire parameters from user #####

feature = input("Enter feature model: ")
while not (feature in ["color", "lbp", "hog"]):
  print("Invalid input - choose from \"color\", \"lbp\", \"hog\"")
  feature = input("Enter feature model: ")

img_type = input("Enter image type: ")
while not (img_type in database):
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

if(feature == 'color'):
  feature_extracted_dataset = np.zeros((40,10,3*64))
elif(feature == 'lbp'):
  feature_extracted_dataset = np.zeros((40,10,256))
else:
  feature_extracted_dataset = np.zeros((40,10,1764))

for x in range(40):
  for y in range(10):
    if(type(database[img_type][x][y]) == type(np.zeros(0))):
      feature_extracted_dataset[x][y] = extract_feature(feature, database[img_type][x][y])    
    else:
      print("None Image Fault: Discarding image-"+ str(x) + "-" + str(y))

# Going from 40x10xnxm to 400x(n*m)
if(feature == 'color'):
  data_set = feature_extracted_dataset.reshape(400, 3*64)
elif(feature == 'lbp'):
  data_set = feature_extracted_dataset.reshape(400, 256)
elif(feature == 'hog'):
  data_set = feature_extracted_dataset.reshape(400, 1764)


latent_features_dataset = dimensionality_reduction(data_set, k, reduction_technique)

# Going from 400xk to 40xk by averaging 10 samples/subject
subject_weight_pairs = np.zeros((40,k))
for subject_id in range(40):
  for feature_i in range(k):
    sum = 0
    for sample_id in range(10):
      sum += latent_features_dataset[(subject_id*10 + sample_id)][feature_i]
    subject_weight_pairs[subject_id][feature_i] = sum/10

# -> OUTPUT = subject_weight_pairs (40 x k)
