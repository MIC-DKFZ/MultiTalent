from batchgenerators.utilities.file_and_folder_operations import *
import shutil
import numpy as np
from multitalent.dataset_conversion.generate_dataset_json import generate_dataset_json
from multitalent.paths import nnUNet_raw
print(nnUNet_raw)



def convert_ms(taskname: str, basedir: str, nnunet_dataset_id: int,):
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

    train_dir = join(basedir, "Training")
    val_dir = join(basedir, "Validation")
    count = 0
    for patient in os.listdir(train_dir):
        d_path = join(train_dir, patient)
        if isdir(d_path):
            count +=1
            shutil.copy(join(d_path, patient +  '_gt.nii.gz'), join(imagestr, patient + '_0000.nii.gz'))
            shutil.copy(join(d_path, patient + '_label.nii.gz'), join(labelstr, patient + '.nii.gz'))
    for patient in os.listdir(val_dir):
        d_path = join(val_dir, patient)
        if isdir(d_path):
            shutil.copy(join(d_path, patient +  '_gt.nii.gz'), join(imagests, patient + '_0000.nii.gz'))



    generate_dataset_json(out_base, {0: "LGE-MRI"},
                          labels={
                              "background": 0,
                              "Right Atrium Cavity": 1,
                              "Left Atrium Cavity": 2,
                              "Left & Right Atrium Wall": 3
                          },
                          num_training_cases=count, file_ending='.nii.gz',
                          dataset_name=task_name, reference='mbas2024 vhallenge',
                          release='release',
                          overwrite_image_reader_writer='NibabelIOWithReorient',
                          description="")



inpath = '/home/constantin/E132-Projekte/Projects/2024_Ulrich_Baumgartner_Mbas_challenge/MBAS_Dataset'
taskname = 'MBAS'
convert_ms(taskname, inpath, 401)