import os

from batchgenerators.utilities.file_and_folder_operations import *
import shutil
import SimpleITK as sitk
from multitalent.dataset_conversion.generate_dataset_json import generate_dataset_json
from multitalent.paths import nnUNet_raw
print(nnUNet_raw)
import csv



def convert_atlas(taskname: str, basedir: str, img_dir: str,label_dict:dict):

    foldername = basedir

    # setting up nnU-Net folders
    out_base = foldername
    imagestr = join(out_base, "imagesTr")
    labelstr = join(out_base, "labelsTr")
    imagests = join(out_base, "imagesTs")
    labelsts = join(out_base, "labelsTs")

    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelsts)

    images = os.listdir(img_dir)
    labels = os.listdir(labelstr)
    for file in labels:
        id_1 = file.split('_')[1]
        for case in images:
            if case.endswith('0000.nii.gz'):
                 if id_1 in case:
                     img = sitk.ReadImage(join(img_dir, case))
                     label = sitk.ReadImage(join(labelstr, file))
                     if img.GetSize() == label.GetSize() and img.GetSpacing() == label.GetSpacing():
                         if isfile(join(imagestr, file[:-7] + '_0000.nii.gz')):
                             print(file)
                         shutil.copy(join(img_dir, case), join(imagestr, file[:-7] + '_0000.nii.gz'))



    generate_dataset_json(out_base, {0: "CT"},
                          labels=label_dict,
                          num_training_cases=len(labels), file_ending='.nii.gz',
                          dataset_name=taskname, reference='',
                          release='',
                          description="")


# already have files labelsTr -> need to select and copy fitting autopet images
inpath = join(nnUNet_raw,'Dataset656_Atlas')

# loads label csv to dict
with open('/home/constantin/cluster-checkpoints/label_name.csv', mode='r') as infile:
    reader = csv.reader(infile)
    with open('coors_new.csv', mode='w') as outfile:
        writer = csv.writer(outfile)
        label_dict = {rows[1]:rows[0]  for rows in reader if rows[0]!= 'Label'}

# print(label_dict)
#thats the path to  autopet2 images in nnunetraw format
autopet_CT_nifti = '/home/constantin/E132-Projekte/Projects/2023_Ulrich_bfmultitalent/nnUNet_raw/nnUNet_raw_data/Task610_autopet_ct/imagesTr'
taskname = 'Atlas'
convert_atlas(taskname, inpath,autopet_CT_nifti , label_dict)