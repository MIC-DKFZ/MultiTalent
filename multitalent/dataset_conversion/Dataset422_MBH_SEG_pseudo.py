from batchgenerators.utilities.file_and_folder_operations import *
import shutil
import numpy as np
from multitalent.dataset_conversion.generate_dataset_json import generate_dataset_json
from multitalent.paths import nnUNet_raw
print(nnUNet_raw)



def convert_ms(taskname: str, basedir: str, nnunet_dataset_id: int,pseudo_dir: str):
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
    for patient in os.listdir(join(pseudo_dir, 'anybleed')):
        if patient.endswith('.nii.gz'):
            count += 1
            shutil.copy(join(pseudo_dir, 'anybleed', patient), join(imagestr, patient))
            shutil.copy(join(pseudo_dir, 'anybleed_pred', patient[:-12] + '.nii.gz'), join(labelstr, patient[:-12] +'.nii.gz'))

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
pseudo_dir = '/home/constantin/E132-Projekte/Projects/2024_ulrich_mbhseg'
#convert_ms(taskname, inpath, 422, pseudo_dir)

#special split for pseudo files not in val

split = load_json('/home/constantin/cluster-data/multitalent/nnUNet_preprocessed/Dataset402_MBH_SEG/splits_final.json')
for fold in split:
    for file in os.listdir('/home/constantin/E132-Projekte/Projects/2024_ulrich_mbhseg/anybleed_pred'):
        if file.endswith('.gz'):
            fold['train'].append(file[:-7])
save_json(split, '/home/constantin/cluster-data/multitalent/nnUNet_preprocessed/Dataset422_MBH_SEG_pseudo/splits_final.json')

