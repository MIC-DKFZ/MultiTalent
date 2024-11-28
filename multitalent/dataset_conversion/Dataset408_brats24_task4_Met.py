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
    valdir = '/home/constantin/E132-Projekte/Projects/2024_Ulrich_beyond_brats/MICCAI-BraTS2024-MET-Challenge-ValidationData/MICCAI-BraTS2024-MET-Challenge-Validation'
    for patient in os.listdir(valdir):
        if patient.startswith('BraTS-MET'):
            t1_path = join(valdir,patient,patient +'-t1n.nii.gz')
            t1c_path = join(valdir,patient,patient +'-t1c.nii.gz')
            t2_path = join(valdir,patient,patient +'-t2w.nii.gz')
            t2f_path = join(valdir,patient,patient +'-t2f.nii.gz')


            shutil.copy(t1_path, join(imagests, patient + '_0000.nii.gz'))
            shutil.copy(t1c_path, join(imagests, patient + '_0001.nii.gz'))
            shutil.copy(t2_path, join(imagests, patient + '_0002.nii.gz'))
            shutil.copy(t2f_path, join(imagests, patient + '_0003.nii.gz'))

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
    nii_files = nifti_files(labelsTr_folder, join=False)
    patients = np.unique([i[:len('BraTS-MET-00822')] for i in nii_files])
    rs = np.random.RandomState(seed)
    rs.shuffle(patients)
    splits = []
    for fold in range(5):
        val_patients = patients[fold::5]
        train_patients = [i for i in patients if i not in val_patients]
        val_cases = [i[:-7] for i in nii_files for j in val_patients if i.startswith(j)]
        train_cases = [i[:-7] for i in nii_files for j in train_patients if i.startswith(j)]
        splits.append({'train': train_cases, 'val': val_cases})
    return splits


inpath = '/home/constantin/E132-Projekte/Projects/2024_Ulrich_beyond_brats/MICCAI-BraTS2024-MET-Challenge-TrainingData_1&2'
taskname = 'brats24_task4_Met'
convert_brats(taskname, inpath, 408)
dataset_name = 'Dataset408_' + taskname
preprocessed_folder = join(nnUNet_preprocessed, dataset_name)
maybe_mkdir_p(preprocessed_folder)
split = create_brats_4_split(join(nnUNet_raw, dataset_name, 'labelsTr'))
save_json(split, join(preprocessed_folder, 'splits_final.json'), sort_keys=False)