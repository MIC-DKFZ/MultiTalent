import os

from batchgenerators.utilities.file_and_folder_operations import *
import shutil
import SimpleITK as sitk
from multitalent.dataset_conversion.generate_dataset_json import generate_dataset_json
from multitalent.paths import nnUNet_raw
print(nnUNet_raw)
import csv

label_dict = {
    "background": 0,
    "liver": 1,
    "spleen": 2,
    "left_kidney": 3,
    "right_kidney": 4,
    "stomach": 5,
    "gallbladder": 6,
    "esophagus": 7,
    "pancreas": 8,
    "duodenum": 9,
    "colon": 10,
    "intestine": 11,
    "adrenal": 12,
    "rectum": 13,
    "bladder": 14,
    "Head_of_femur_L": 15,
    "Head_of_femur_R": 16
}

in_path = '/home/constantin/cluster-data/multitalent/data_multitalent/Dataset654_word'
cases = len(os.listdir(join(in_path, 'imagesTr')))
generate_dataset_json(in_path, {0: "CT"},
                      labels=label_dict,
                      num_training_cases=cases, file_ending='.nii.gz',
                      dataset_name='word', reference='',
                      release='',
                      description="")