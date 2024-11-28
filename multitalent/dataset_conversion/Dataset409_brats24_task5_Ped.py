from batchgenerators.utilities.file_and_folder_operations import *
import shutil
import numpy as np
from multitalent.dataset_conversion.generate_dataset_json import generate_dataset_json
from multitalent.paths import nnUNet_raw
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
        if patient.startswith('BraTS-PED'):
            count +=1
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
                              "ET": 1,
                              "NET": 2,
                              "CC": 3,
                                "ED": 4},
                          num_training_cases=count, file_ending='.nii.gz',
                          dataset_name=task_name, reference='brats24_task5_Ped',
                          release='release',
                          overwrite_image_reader_writer='SimpleITKIO',
                          description="brats",
                          license='')



inpath = '/omics/groups/OE0441/E132-Projekte/Projects/2024_Ulrich_beyond_brats/BraTS2024-PED-Challenge-TrainingData/BraTS-PEDs2024_Training'
taskname = 'brats24_task5_Ped'
convert_brats(taskname, inpath, 409)