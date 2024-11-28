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

    train_dir = join(basedir, "label_192")
    val_dir = join(basedir, "unlabel_2000")
    count = 0
    d_path = join(train_dir, 'images')
    gt_path = join(train_dir, 'ground truths')
    for patient in os.listdir(d_path):
        if patient.endswith('.nii.gz'):
            count +=1
            shutil.copy(join(d_path, patient), join(imagestr, patient[:-7] + '_0000.nii.gz'))
            shutil.copy(join(gt_path, patient), join(labelstr, patient ))

    generate_dataset_json(out_base, {0: "CT"},
                          labels={
                              "background": 0,
                              "epidural": 1,
                              "intraparenchymal": 2,
                              "intraventricular": 3,
                              "subarachnoid": 4,
                              "subdural": 5
                          },
                          num_training_cases=count, file_ending='.nii.gz',
                          dataset_name=task_name, reference='2024_MHB_SEG',
                          release='release',
                          overwrite_image_reader_writer='SimpleITKIO',
                          description="bleedbrain",
                          license='AIML')



inpath = '/home/constantin/E132-Rohdaten/Challenges/2024_MBH_SEG/'
taskname = 'MBH_SEG'
convert_ms(taskname, inpath, 402)