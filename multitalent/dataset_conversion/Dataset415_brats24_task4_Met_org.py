import os

from batchgenerators.utilities.file_and_folder_operations import *
import shutil
import numpy as np
from multitalent.dataset_conversion.generate_dataset_json import generate_dataset_json
from multitalent.paths import nnUNet_raw, nnUNet_preprocessed

print(nnUNet_raw)



def convert_brats(taskname: str, basedir: str, nnunet_dataset_id: int,):
    task_name = taskname

    foldername = "Dataset%03.0d_%s" % (nnunet_dataset_id, task_name)

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

    count = 0

    for patient in os.listdir(basedir):
        if patient.startswith('BraTS-MET'):

            count +=1
            print(count)
            t1_path = join(basedir,patient,patient +'-t1n.nii.gz')
            t1c_path = join(basedir,patient,patient +'-t1c.nii.gz')
            t2_path = join(basedir,patient,patient +'-t2w.nii.gz')
            t2f_path = join(basedir,patient,patient +'-t2f.nii.gz')
            seg_path = join(basedir,patient,patient +'-seg.nii.gz')

            shutil.copy(t1_path, join(imagestr, patient + '_0000.nii.gz'))
            shutil.copy(t1c_path, join(imagestr, patient + '_0001.nii.gz'))
            shutil.copy(t2_path, join(imagestr, patient + '_0002.nii.gz'))
            shutil.copy(t2f_path, join(imagestr, patient + '_0003.nii.gz'))
            shutil.copy(seg_path, join(labelstr, patient + '.nii.gz'))




    generate_dataset_json(out_base, {"0": "T1", "1": "T1ce", "2": "T2", "3": "Flair"},
                          labels={
                              "background": 0,
                              "NETC": 1,
                              "SNFH": 2,
                              "ET": 3},
                          num_training_cases=count, file_ending='.nii.gz',
                          dataset_name=task_name, reference='brats24_task4_Met',
                          release='release',
                          overwrite_image_reader_writer='SimpleITKIO',
                          description="brats",
                          license='')

def create_brats_4_split(labelsTr_folder: str, seed: int = 1234) -> List[dict[str, List]]:
    old_splits= load_json('/home/constantin/cluster-data/multitalent/nnUNet_preprocessed/Dataset408_brats24_task4_Met/splits_final.json')
    splits = []
    all_images = os.listdir(join(nnUNet_raw, 'Dataset415_brats24_task4_Met_org', 'labelsTr'))
    for fold in range(5):
        train_cases = []
        val_cases = []
        for tr in old_splits[fold]['train']:
            if tr + '.nii.gz' in all_images:
                train_cases.append(tr)
        for val in old_splits[fold]['val']:
            if val + '.nii.gz' in all_images:
                val_cases.append(val)
        splits.append({'train': train_cases, 'val': val_cases})
    return splits


inpath = '/home/constantin/E132-Projekte/Projects/2024_Ulrich_beyond_brats/MICCAI-BraTS2024-MET-Challenge-TrainingData_1/MICCAI-BraTS2024-MET-Challenge-Training_1'
taskname = 'brats24_task4_Met_org'
#convert_brats(taskname, inpath, 415)
dataset_name = 'Dataset415_' + taskname
preprocessed_folder = join(nnUNet_preprocessed, dataset_name)
maybe_mkdir_p(preprocessed_folder)
split = create_brats_4_split(join(nnUNet_raw, dataset_name, 'labelsTr'))
save_json(split, join(preprocessed_folder, 'splits_final.json'), sort_keys=False)