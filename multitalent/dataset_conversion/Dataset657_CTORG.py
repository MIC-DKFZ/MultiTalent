import os

from batchgenerators.utilities.file_and_folder_operations import *
import shutil
import SimpleITK as sitk
from multitalent.dataset_conversion.generate_dataset_json import generate_dataset_json
from multitalent.paths import nnUNet_raw
print(nnUNet_raw)
import csv



def convert_atlas(taskname: str, basedir: str, label_dict:dict, id:int):


    # setting up nnU-Net folders
    out_base = join(nnUNet_raw, taskname)
    imagestr = join(out_base, "imagesTr")
    labelstr = join(out_base, "labelsTr")
    imagests = join(out_base, "imagesTs")
    labelsts = join(out_base, "labelsTs")

    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelsts)

    al_img = os.listdir(basedir)
    c=0
    for file in al_img:
        if file.endswith('nii.gz'):
            if file.startswith('volume'):
                shutil.copy(join(inpath,file), join(imagestr,file[7:-7] +'_0000.nii.gz' ))
                c+=1
            if file.startswith('labels'):
                shutil.copy(join(inpath, file), join(labelstr, file[7:]))

    generate_dataset_json(out_base, {0: "CT"},
                          labels=label_dict,
                          num_training_cases=c, file_ending='.nii.gz',
                          dataset_name=taskname, reference='',
                          release='',
                          description="")


# already have files labelsTr -> need to select and copy fitting autopet images
inpath = join('/home/constantin/Downloads/PKG - CT-ORG/CT-ORG/OrganSegmentations')

label_dict ={
    "background": 0,
    "Liver": 1,
    "Bladder": 2,
    "Lungs": 3,
    "Kidneys": 4,
    "Bone": 5,
    "Brain": 6
}
# loads label csv to dict
taskname = 'Dataset657_CTORG'
convert_atlas(taskname, inpath, label_dict, id=657)