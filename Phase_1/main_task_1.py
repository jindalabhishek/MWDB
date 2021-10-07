from model import COLOR_MOMENT
from model import ELBP
from model import HOG
from Util.dao_util import DAOUtil
from feature_descriptor_util import get_color_moment_feature_descriptor
from feature_descriptor_util import get_elbp_feature_descriptor
from feature_descriptor_util import get_hog_feature_descriptor


def main():
    """
        Executes Task-1. Please execute Task-0 before running Task-1.
    """
    print('Welcome to Task 1 Demo. This demo requires 2 inputs from user: Image Id and Model Name')
    print('Please enter the Image Id. The Image Id is defined by label (0-39) concatenated with "_" and variant of the '
          'corresponding image (1-10).')
    image_id = str(input('Enter the Image Id:'))
    print('You Entered Image Id:', image_id)
    """
        Loads the image object (label, pixels) from MongoDB based on image Id
    """
    image_object = DAOUtil().get_image_for_label(image_id)
    image_pixels = image_object.get('image_pixels')
    model_name = str(input('Enter the model name out of (cm8x8, elbp, hog):'))
    """
        Prints Feature Descriptor Based on Model name input
    """
    if model_name == COLOR_MOMENT:
        color_moment_feature_descriptor = get_color_moment_feature_descriptor(image_pixels)
        print('This is color moment feature descriptor of size 8x8x3. The mean, standard deviation '
              'and skewness are stored in the last dimension of the matrix')
        print('    Mean, Standard Deviation, Skewness')
        print(color_moment_feature_descriptor)
    elif model_name == ELBP:
        elbp_feature_descriptor = get_elbp_feature_descriptor(image_pixels)
        print(elbp_feature_descriptor)
    elif model_name == HOG:
        hog_feature_descriptor = get_hog_feature_descriptor(image_pixels)
        print(hog_feature_descriptor)
    else:
        """
        Raise Exception in case model name doesn't match out of valid values
        """
        raise Exception('InvalidInputException: Please enter the correct value for model name')


main()
