import numpy as np
import cv2
import glob
import os  
import sklearn.decomposition as sk_decomp
import skimage.feature as skf
import math




def color_moments(img):	# Calculate 1st, 2nd, 3rd color moments
  # Input image format - (64x64)
  # Return format - (3x64)
  m1 = []
  m2 = []
  m3 = []
  img_split = image_split(img)
  for i in range(64):
    sum = 0.
    for x in range(8):
      for y in range(8):
        sum = sum + img_split[i][x][y]
    box_ave = sum/64
    m1.append(sum/64)

    m2_sum = 0.
    m3_sum = 0.
    for x in range(8):
      for y in range(8):
        m2_sum += pow(img_split[i][x][y] - box_ave, 2)
        m3_sum += pow(img_split[i][x][y] - box_ave, 3)
    m2.append(math.sqrt(m2_sum/64))
    m3.append(np.cbrt(m3_sum/64))

  return np.array((np.array(m1),np.array(m2),np.array(m3))).flatten()

def lbp_extract(img):
  lbp = skf.local_binary_pattern(img, 8, 1, method="var")
  lbp = lbp.flatten()
  hist,_ = np.histogram(lbp, bins=256, range=(0,255))
  return hist

def hog_extract(img):	# skimage.hog wrapper
  ret_out, ret_hog = skf.hog(img, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), block_norm='L2-Hys', visualize=True)
  return ret_out

def extract_feature(model, img):
  if(model == "color"):
    return color_moments(img)
  elif(model == "lbp"):
    return lbp_extract(img)
  elif(model == "hog"):
    return hog_extract(img)
  else:
    print("Invalid model - Choose from: \"color\", \"lbp\", \"hog\"")
    return "Fault"



def normalize_data_for_lda(image_vector_matrix):
    normalized_data = (image_vector_matrix - np.min(image_vector_matrix)) \
                      / (np.max(image_vector_matrix) - np.min(image_vector_matrix))
    return normalized_data

def compute(data, k, image_types, *args):
    lda = sk_decomp.LatentDirichletAllocation(n_components=k, random_state=0)
    latent_features = lda.fit_transform(normalize_data_for_lda(data), image_types)
    # self.latent_features = latent_features
    # self.input_matrix = data
    # self.objects_in_k_dimensions = latent_features.real
    return latent_features.real


def retrive_data(path,model_name,k_d):
	os.chdir(path)  

	all_image=[]
	all_labels=[[],[],[],[]]
	# path='/home/zaid/Documents/ASU/1000/'       
	for file in list(glob.glob('*.png')):
	 	
	 	a=cv2.imread(path+file,0)
	 	all_image.append(a)
	 	all_labels[0].append(file[:-4].split('-')[1])
	 	all_labels[1].append(int(file[:-4].split('-')[2]))
	 	all_labels[2].append(int(file[:-4].split('-')[3]))
	 	all_labels[3].append(file)



	all_feature_lbp=[]


	if model_name=='CM':
		for img in all_image:
			all_feature_lbp.append(extract_feature("color",img))

	if model_name=='ELBP':
		for img in all_image:
			all_feature_lbp.append(extract_feature("lbp",img))

	if model_name=='HOG':
		for img in all_image:
			all_feature_lbp.append(extract_feature("hog",img))

		
	return compute(np.asarray(all_feature_lbp),k_d,all_labels[3]), all_labels


# k,l=(retrive_data('/home/zaid/Documents/ASU/1000/','lda'))
# print(l[3][:15])




