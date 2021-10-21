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


def task_3():
  
  database = read_data_type()

  ##### Acquire parameters from user #####

  feature = input("Enter feature model: ")
  while not (feature in ["color", "lbp", "hog"]):
    print("Invalid input - choose from \"color\", \"lbp\", \"hog\"")
    feature = input("Enter feature model: ")

  img_type = input("Enter image subject: ")
  while not (img_type in database):
    print("Invalid input - choose subject id from \"1 to 40\"")
    img_type = input("Enter image type: ")

  ##### Extract and preprocess data matrix #####

  if(feature == 'color'):
    feature_extracted_dataset = np.zeros((13,10,3*64))
  elif(feature == 'lbp'):
    feature_extracted_dataset = np.zeros((13,10,256))
  else:
    feature_extracted_dataset = np.zeros((13,10,1764))

  p=1
  for x in range(13):
    for y in range(10):
      if (type(database[img_type][x][y]) == type(np.zeros(0))):
        feature_extracted_dataset[x][y] = extract_feature(feature, database[img_type][x][y])    
      else:
        if p==1:
          print("None Image Fault: Discarding image-"+ str(x) + "-" + str(y), end = ', ')
          p=2
        else:print("image-"+ str(x) + "-" + str(y), end = ', ')

  else:print('\n')
  # Going from 40x10xnxm to 400x(n*m)
  if(feature == 'color'):
    data_set = feature_extracted_dataset.reshape(130, 3*64)
  elif(feature == 'lbp'):
    data_set = feature_extracted_dataset.reshape(130, 256)
  elif(feature == 'hog'):
    data_set = feature_extracted_dataset.reshape(130, 1764)


  type_weights = []
  for i in range(0,data_set.shape[0],10):
    p = data_set[i:i+10]
    type_weights.append(p.mean(axis=0))
  type_weights=np.array(type_weights)

  print("Calculating Similarity Matrix")
  
  similarity = similar_matrix(type_weights)

  k = int(input("Enter k value: "))

  reduction_technique = input("Enter dimensionality reduction technique: ")
  while not (reduction_technique in ["pca", "svd", "lda", "kmean"]):
    print("Invalid input - choose from \"pca\", \"svd\", \"lda\", \"kmean\"")
    reduction_technique = input("Enter dimensionality reduction technique: ")
  
  latent_type_features_dataset = dimensionality_reduction(abs(similarity), k, reduction_technique)

  k= '/content/ok' + '/TypeType_similarity_features'+str(datetime.now())+'.json'
  with open (k, 'w') as outfile:
                json.dump(latent_type_features_dataset, outfile, indent=2)
  
  return latent_type_features_dataset, k

def task_4():

  database = read_data_subject()

  ##### Acquire parameters from user #####

  feature = input("Enter feature model: ")
  while not (feature in ["color", "lbp", "hog"]):
    print("Invalid input - choose from \"color\", \"lbp\", \"hog\"")
    feature = input("Enter feature model: ")

  img_type = input("Enter image type: ")
  while not (img_type in database):
    print("Invalid input - choose from \"cc\", \"con\", \
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
  p=1
  for x in range(40):
    for y in range(10):
      if(type(database[img_type][x][y]) == type(np.zeros(0))):
        feature_extracted_dataset[x][y] = extract_feature(feature, database[img_type][x][y])    
      else:
        if p==1:
          print("None Image Fault: Discarding image-"+ str(x) + "-" + str(y), end = ', ')
          p=2
        else:print("image-"+ str(x) + "-" + str(y), end = ', ')

  # Going from 40x10xnxm to 400x(n*m)
  if(feature == 'color'):
    data_set = feature_extracted_dataset.reshape(400, 3*64)
  elif(feature == 'lbp'):
    data_set = feature_extracted_dataset.reshape(400, 256)
  elif(feature == 'hog'):
    data_set = feature_extracted_dataset.reshape(400, 1764) 


  type_weights = []
  for i in range(0,data_set.shape[0],10):
    p = data_set[i:i+10]
    type_weights.append(p.mean(axis=0))
  type_weights=np.array(type_weights)

  similarity = similar_matrix(type_weights)

  print('calulating Similarity Matrix')
  similarity = similar_matrix(np.array(abs(type_weights)))

  k = int(input("Enter k value: "))

  reduction_technique = input("Enter dimensionality reduction technique: ")
  while not (reduction_technique in ["pca", "svd", "lda", "kmean"]):
    print("Invalid input - choose from \"pca\", \"svd\", \"lda\", \"kmean\"")
    reduction_technique = input("Enter dimensionality reduction technique: ")
  
  latent_type_features_dataset = dimensionality_reduction((similarity), k, reduction_technique)
  
  k= '/content/ok' + '/SubSub_similarity_features'+str(datetime.now())+'.json'
  with open (k, 'w') as outfile:
                json.dump(latent_type_features_dataset, outfile, indent=2)
  
  return latent_type_features_dataset, k
[22:57, 20/10/2021] Kirity Asu Mwdb: 