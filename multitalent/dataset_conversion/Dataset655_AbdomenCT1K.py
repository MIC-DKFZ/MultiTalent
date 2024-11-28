import os

from batchgenerators.utilities.file_and_folder_operations import *
import shutil
import numpy as np
from multitalent.dataset_conversion.generate_dataset_json import generate_dataset_json
from multitalent.paths import nnUNet_raw
print(nnUNet_raw)



def convert_abdomen1k(taskname: str, basedir: str, nnunet_dataset_id: int,):

    foldername = basedir

    # setting up nnU-Net folders
    out_base = join(nnUNet_raw, foldername)
    imagestr = join(out_base, "imagesTr")
    labelstr = join(out_base, "labelsTr")
    imagests = join(out_base, "imagesTs")
    labelsts = join(out_base, "labelsTs")

    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelsts)

    images = os.listdir(imagestr)
    labels = os.listdir(labelstr)
    for file in images:
        if not file[:-12] + '.nii.gz' in labels:
            shutil.move(join(imagestr, file), join(imagests, file))

    generate_dataset_json(out_base, {0: "CT"},
                          labels={
                              "background": 0,
                              "liver": 1,
                              "kidney": 2,
                              "spleen": 3,
                              "pancreas": 2

                          },
                          num_training_cases=len(labels), file_ending='.nii.gz',
                          dataset_name=taskname, reference='',
                          release='',
                          description="")


# already have files in imageTr and labelsTr -> but numbers is not the same (more images)
inpath = join(nnUNet_raw,'Dataset655_AbdomenCT1K')
taskname = 'AbdomenCT1K'
convert_abdomen1k(taskname, inpath, 655)