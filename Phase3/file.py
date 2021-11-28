import numpy as np
import cv2
import glob
import os
import sklearn.decomposition as sk_decomp
import skimage.feature as skf
import math

from sklearn.metrics import multilabel_confusion_matrix

import vector_util
import feature_descriptor_util
from dimensionality_reduction.SVD import SVD
from dimensionality_reduction.PCA import PCA
from dimensionality_reduction.LDA import LDA
from dimensionality_reduction.KMeans import KMeans
from Phase_1.Constants import GREY_SCALE_MAX
import image_comparison_util


def color_moments(img):  # Calculate 1st, 2nd, 3rd color moments
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
        box_ave = sum / 64
        m1.append(sum / 64)

        m2_sum = 0.
        m3_sum = 0.
        for x in range(8):
            for y in range(8):
                m2_sum += pow(img_split[i][x][y] - box_ave, 2)
                m3_sum += pow(img_split[i][x][y] - box_ave, 3)
        m2.append(math.sqrt(m2_sum / 64))
        m3.append(np.cbrt(m3_sum / 64))

    return np.array((np.array(m1), np.array(m2), np.array(m3))).flatten()


def image_split(img):  # Split 64x64 image into 64x8x8 image
    # Image format - (64x64)
    # Return format - (64x8x8)

    ret_img = np.zeros((64, 8, 8))
    for i in range(64):
        for x in range(8):
            for y in range(8):
                xo = (i * 8 + x) % 64
                yo = int(i / 8) * 8 + y
                ret_img[i][x][y] = img[xo][yo]
    return ret_img


def color_moments(img):  # Calculate 1st, 2nd, 3rd color moments
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
        box_ave = sum / 64
        m1.append(sum / 64)

        m2_sum = 0.
        m3_sum = 0.
        for x in range(8):
            for y in range(8):
                m2_sum += pow(img_split[i][x][y] - box_ave, 2)
                m3_sum += pow(img_split[i][x][y] - box_ave, 3)
        m2.append(math.sqrt(m2_sum / 64))
        m3.append(np.cbrt(m3_sum / 64))

    return np.array((np.array(m1), np.array(m2), np.array(m3))).flatten()


def lbp_extract(img):
    lbp = skf.local_binary_pattern(img, 24, 8, method="uniform")
    num_points = 24
    (hist, _) = np.histogram(lbp.ravel(), bins=range(0, num_points + 3),
                             range=(0, num_points + 2))

    eps = 1e-7
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)
    return hist

def hog_extract(img):  # skimage.hog wrapper
    ret_out, ret_hog = skf.hog(img, orientations=9,
                               pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True,
                               feature_vector=True)
    return ret_out


def extract_feature(model, img):
    if model == "color":
        return color_moments(img)
    elif model == "lbp":
        return lbp_extract(img)
    elif model == "hog":
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


def getImageData(path,model_name,labelFunc):
    all_image = []
    all_labels = []
    fileNames = []
    for file in os.listdir(path):
        a = vector_util.convert_image_to_matrix(os.path.join(path, file))
        a = a / GREY_SCALE_MAX
        all_image.append(a)
        all_labels.append(labelFunc(file))
        fileNames.append(file)

    all_feature_lbp = []

    if model_name == 'CM':
        for img in all_image:
            all_feature_lbp.append(feature_descriptor_util \
                .get_reshaped_color_moment_vector(
                feature_descriptor_util.get_color_moment_feature_descriptor(img)))

    if model_name == 'ELBP':
        for img in all_image:
            all_feature_lbp.append(
                image_comparison_util.get_elbp_histogram(feature_descriptor_util.get_elbp_feature_descriptor(img)))

    if model_name == 'HOG':
        for img in all_image:
            all_feature_lbp.append(feature_descriptor_util.get_hog_feature_descriptor(img))
    return all_feature_lbp,all_labels, fileNames

def getTrainData(path, model_name, k_d,labelFunc):
    all_feature_lbp, all_labels, fileNames = getImageData(path,model_name,labelFunc)
    dimensionality_reduction = KMeans(1000)
    objects = dimensionality_reduction.compute(np.array(all_feature_lbp), k_d, all_labels)
    return dimensionality_reduction, fileNames

def getTestData(path,model_name,dimension_reduction,labelFunc):
    all_feature_lbp, all_labels, fileNames = getImageData(path,model_name,labelFunc)
    all_vals = []
    for i in dimension_reduction.transform(np.array(all_feature_lbp)):
        all_vals.append(i/np.linalg.norm(i))
    return dimension_reduction.transform(np.array(all_feature_lbp)), all_labels, fileNames
# k,l=(retrive_data('/home/zaid/Documents/ASU/1000/','lda'))
# print(l[3][:15])


def calculate_and_print_results(Y_test, Y_hat, num2type):
    cm = multilabel_confusion_matrix(Y_test, Y_hat, labels=list(num2type.values()))
    fp = {}
    misses = {}
    tp = {}
    tn = {}

    fp_rate = {}
    miss_rate = {}

    total_fp, total_fn, total_tp, total_tn = 0, 0, 0, 0
    total_misses = 0
    for i in range(len(num2type.values())):
        fp[list(num2type.values())[i]] = cm[i][0][1]
        misses[list(num2type.values())[i]] = cm[i][1][0]
        tp[list(num2type.values())[i]] = cm[i][1][1]
        tn[list(num2type.values())[i]] = cm[i][0][0]

        fp_rate[list(num2type.values())[i]] = fp[list(num2type.values())[i]] / (fp[list(num2type.values())[i]] + tn[list(num2type.values())[i]])
        total_fp += cm[i][0][1]
        total_tp += cm[i][1][1]
        total_tn += cm[i][0][0]
        total_misses += cm[i][1][0]

    for i in range(len(num2type.values())):
        miss_rate[list(num2type.values())[i]] = misses[list(num2type.values())[i]] / total_misses
    print('Total false positives = ', total_fp)
    print(fp_rate)
    print('Total misses = ', total_misses)
    print(miss_rate)
    print('Total correctly classified = ', total_tp)
    print(tp)
