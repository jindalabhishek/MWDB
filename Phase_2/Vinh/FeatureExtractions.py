import skimage.feature as skf
import numpy as np
import math

def image_split(img):	# Split 64x64 image into 64x8x8 image
  # Image format - (64x64)
  # Return format - (64x8x8)

  ret_img = np.zeros((64,8,8))
  for i in range(64):
    for x in range(8):
      for y in range(8):
        xo = (i*8 + x)%64
        yo = int(i/8)*8 + y
        ret_img[i][x][y] = img[xo][yo]
  return ret_img

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
