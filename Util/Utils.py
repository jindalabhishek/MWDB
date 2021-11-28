import json
from Phase_1.feature_descriptor_util import *
from Phase_1.image_comparison_util import *
from json_util import LatentSemanticFile

OUTPUT_FILE_DIR = '../Outputs/'


def get_output_file_path(task_id, feature_model, topic, reduction_technique):
    full_path = OUTPUT_FILE_DIR + task_id
    full_path_list = [full_path, reduction_technique]
    if feature_model:
        full_path_list.append(feature_model)
    if topic:
        full_path_list.append(topic)
    return "_".join(full_path_list) + ".json"


def save_task_data(task_id, dimension_reduction_object, task_output=None, topic=None, feature_model=None):
    technique_name = type(dimension_reduction_object).__name__
    output_path = get_output_file_path(task_id, feature_model, topic, technique_name)
    print(output_path)
    LatentSemanticFile(feature_model, dimension_reduction_object, task_output, task_id).serialize(output_path)


def get_task_output_data(input_path):
    return LatentSemanticFile.deserialize(input_path).task_output


def get_json_data(input_path):
    return LatentSemanticFile.deserialize(input_path)


def sort_feature_weight_pair(feature_weight_pair_dict):
    feature_weight_pair = [[t, feature_weight_pair_dict[t]] for t in feature_weight_pair_dict]
    return sorted(feature_weight_pair, key=lambda x: sum([i * i for i in x[1]]))


def get_similarity_matrix(input_path):
    file_object = open(input_path)
    json_data = json.load(file_object)
    latent_feature_json_object = json_data[LatentSemanticFile.LATENT_FEATURES]
    return latent_feature_json_object['matrix_nxm']


def print_k_latent_semantics_in_sorted_weight_pairs(sorted_weight_pair):
    latent_semantic_vs_subject_weights = {}
    for subject_id_vs_latent_semantics in sorted_weight_pair:
        subject_id = subject_id_vs_latent_semantics[0]
        latent_semantics = subject_id_vs_latent_semantics[1]
        for i in range(0, len(latent_semantics)):
            latent_semantic = latent_semantics[i]
            if latent_semantic_vs_subject_weights.get(i):
                latent_semantic_vs_subject_weights.get(i).append([subject_id, latent_semantic])
            else:
                latent_semantic_vs_subject_weights[i] = [[subject_id, latent_semantic]]
    # print(latent_semantic_vs_subject_weights)
    latent_semantic_vs_subject_weights = list(latent_semantic_vs_subject_weights.items())

    def function(x):
        return x[1]

    for k in range(0, len(latent_semantic_vs_subject_weights)):
        latent_semantic_vs_subject_weights[k][1].sort(key=function, reverse=True)

    print('Sorted K latent Semantic File ')
    print(latent_semantic_vs_subject_weights)


def get_query_image_feature_descriptor(feature_model, query_image_vector):
    query_image_feature_descriptor = None
    if feature_model == 'CM':
        color_moment_feature_descriptor = get_color_moment_feature_descriptor(query_image_vector)
        color_moment_feature_descriptor = get_reshaped_color_moment_vector(color_moment_feature_descriptor)
        query_image_feature_descriptor = color_moment_feature_descriptor.copy()
    elif feature_model == 'ELBP':
        elbp_feature_descriptor = get_elbp_feature_descriptor(query_image_vector)
        elbp_feature_descriptor = get_elbp_histogram(elbp_feature_descriptor)
        query_image_feature_descriptor = elbp_feature_descriptor.copy()
    elif feature_model == 'HOG':
        hog_feature_descriptor = get_hog_feature_descriptor(query_image_vector)
        query_image_feature_descriptor = hog_feature_descriptor.copy()
    return query_image_feature_descriptor

def getType(fileName):
    return fileName.split("-")[1]

def getSubject(fileName):
    return fileName.split("-")[2]

def getSample(fileName):
    return fileName.split("-")[3].split(".")[0]

def getNormalDistance(x):
    normal_x = (np.pi*np.std(x)) * np.exp(-0.5*((x-np.mean(x))/np.std(x))**2)
    return normal_x

def getPearsonDistance(x,y):
    a = getNormalDistance(x)
    b = getNormalDistance(y)
    return np.sum(a * b) / np.sqrt(np.sum(np.square(a)) * np.sum(np.square(b)))

def getEuclideanDistance(x,y):
    return np.linalg.norm(x - y)

def getIntersectionSimilarity(x,y):
    return np.std(x/y)

def getChiSquareDistance(x,y):
    return 0.5 * np.sum(((x - y) ** 2) /(x + y + 1e-10))
