from sklearn.decomposition import TruncatedSVD, PCA
import sklearn.decomposition as sk_decomp
import numpy as np

def get_reduced_matrix_using_pca(data, k, cov_method='np'):

  n_objects, n_features = data.shape

  COV = np.zeros((n_features,n_features))
  
  # Calculate COV matrix
  if(cov_method == 'manual'):
    for feature_i in range(n_features):
     feature_i_ave = np.mean(data[:,feature_i])
     for feature_j in range(n_features):
      feature_j_ave = np.mean(data[:,feature_j])
      sum = 0
      for object in range(n_objects):
        sum += (data[object,feature_i] - feature_i_ave)*(data[object,feature_j] - feature_j_ave)
      COV[feature_i, feature_j] = sum/n_objects
  elif(cov_method == 'auto'):
    COV = np.dot(data.T, data)/(n_features-1) # Feature x Feature COV matrix
  elif(cov_method == 'np'):
    COV = np.cov(data, rowvar=False)

  #Find eigenvalues and eigenvectors
  eig_val, eig_vec = np.linalg.eig(COV)

  if(k > eig_val.shape[0]):
    print("ERROR - k is larger than available latent semantics = " + str(eig_val.shape[0]))
    exit()

  # Sort eig_vec by eig_val in descending order
  sorted_eig_val = sorted(eig_val, reverse=True)
  eig_val_list = eig_val.tolist()
  sorted_eig_val_index = [eig_val_list.index(i) for i in sorted_eig_val]  # 0 is highest

  # Choose top-k latent features
  selected_eig_vec = []
  for i in range(k):
    selected_eig_vec.append(eig_vec[sorted_eig_val_index[i]])
  selected_eig_vec = np.transpose(np.array(selected_eig_vec))
  
  # Transform data
  latent_features = np.matmul(data,selected_eig_vec)
  return latent_features

def top_eigen_vectors(eigen_value,eigen_vec, k):
    val = []
    bool_arr = np.isreal(eigen_value)
    for i in range(len(eigen_value)):
        if bool_arr[i] and eigen_value[i]>=0:
            val.append((eigen_value[i], i))
    eigen_value = sorted(val,reverse=True)[:k]
    eigen_vec = eigen_vec.T
    eigen_vec = np.array([eigen_vec[i[1]] for i in eigen_value])
    eigen_vec = eigen_vec.T
    return np.array([i[0] for i in eigen_value]),eigen_vec

def get_reduced_matrix_using_svd(image_vector_matrix, k):
    right_eig_val, right_eig_vec = top_eigen_vectors(*np.linalg.eig(np.dot(image_vector_matrix.T, image_vector_matrix)),k)
    left_eig_val, left_eig_vec = top_eigen_vectors(*np.linalg.eig(np.dot(image_vector_matrix, image_vector_matrix.T)),k)
    return left_eig_vec

def normalize_data_for_lda(image_vector_matrix):
    normalized_data = (image_vector_matrix - np.min(image_vector_matrix)) \
                      / (np.max(image_vector_matrix) - np.min(image_vector_matrix))
    return normalized_data
  
def get_reduced_matrix_using_lda(data, k):
    LDA = sk_decomp.LatentDirichletAllocation(n_components = k, random_state=0)
    normalized_data = normalize_data_for_lda(data)
    latent_features = LDA.fit_transform(normalized_data)
    return latent_features
