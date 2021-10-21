from PIL import Image
import numpy as np
import os

def read_data():
  ret = {'cc': [], 'con': [], 'detail': [], 'emboss': [], 'jitter': [], 'neg': [],\
	'noise01': [], 'noise02': [], 'original': [], 'poster': [], 'rot': [],\
	'smooth': [], 'stipple': []}
  for i in ret:
    subjectID = 40
    sampleID = 10
    arr = [ [[] for col in range(sampleID)] for col in range(subjectID)]
    ret[i] = arr	# X x subjectID x sampleID array

  read_dir = "absolute-directory-to-import-images"
  file_list = os.listdir(read_dir)
  print("Reading images from " + read_dir)
  for i in file_list:
    img_read = Image.open(read_dir+"/"+i)
    img_read = np.array(img_read)
    if(img_read.shape == (64,64)):
      img_name = os.path.basename(i)
      img_name_split = img_name.split("-", 3)
      img_name_split[3] = img_name_split[3].split(".")[0]
      #print("Reading " + img_name_split[0] + "-" + img_name_split[1] + "-" + img_name_split[2] + "-" + img_name_split[3])

      if(img_name_split[1] in ret):
        ret[img_name_split[1]][int(img_name_split[2])-1][int(img_name_split[3])-1] = img_read
      else: print(img_name + " has invalid type - disregarding")
    else: print(img_name + " has invalid format - disregarding")

  return ret

def to_images(db):
  ret = np.zeros((13*40*10, 64, 64))
  set = np.zeros((13,40,10,64,64))
  
  type_count = 0
  for i in db:
    for x in range(40):
      for y in range(10):
        if(type(db[i][x][y]) == type(np.zeros(0))):
          set[type_count][x][y] = db[i][x][y]
        else:
          print("Discarding image " + i + "-" + str(x) + "-" + str(y))
    type_count += 1

  ret = set.reshape((13*400, 64, 64))
  return ret

def to_data(db):
  ret = np.zeros((13*400, 64*64))

  for i in range(13*400):
    ret[i] = db[i].reshape(64*64)

  return ret
  
