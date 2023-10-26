import sys
import copy
sys.path.append('..')
from yolov1 import *


# Dataset path
DATA_PATH = '/home/pushkar/Documents/coding/FreiHand'

# Evaluation directories
FILE_TYPE = 'evaluation'
MASK_PATH = 'segmap'

# For training dataset use the following values
# FILE_TYPE = 'training'
# MASK_PATH = 'mask'

# Creating FreiHand object
evaluation = FreiHand(DATA_PATH, mask_dir=MASK_PATH, file_type=FILE_TYPE)

# loading data files
evaluation.load_data_files()

# Reading the images
evaluation.read_image_files()

# Reading mask files
evaluation.mask_contour()

# Computing the annotations
evaluation.project_landmarks()

# Generate annotations dataframe
# use ``save_csv=True`` to save the csv file
evaluation.generate_annotations(file_name='evaluation_annotations.csv')

# Create a copy of dataframe
evaluation_df = copy.deepcopy(evaluation.annotation_df)

# Mirroring and combining the annotation dataset
evaluation_df_both_hands = mirror_annotations(evaluation_df)

# Saving the new dataset (using ``pandas.DataFrame.to_csv`` method to save the DataFrame)
evaluation_df_both_hands.to_csv("../data/evaluation.csv", index=False)