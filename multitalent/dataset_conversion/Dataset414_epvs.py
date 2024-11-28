from batchgenerators.utilities.file_and_folder_operations import *
import shutil
import numpy as np
from multitalent.dataset_conversion.generate_dataset_json import generate_dataset_json
from multitalent.paths import nnUNet_raw, nnUNet_preprocessed
print(nnUNet_raw)



def convert_epvs(taskname: str, basedir: str, nnunet_dataset_id: int,):
    task_name = taskname

    foldername = "Dataset%03.0d_%s" % (nnunet_dataset_id, task_name)
    val_dir = join(basedir, 'epvs_val/Val/Val')
    basedir = join(basedir, 'epvs_datav3')
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
        if patient.endswith('PVS.nii.gz'):
            case = patient[:-11]
            shutil.copy(join(basedir, case + '_PVS.nii.gz'), join(labelstr, case + '.nii.gz'))
            shutil.copy(join(basedir, case + '_T1w.nii.gz'), join(imagestr, case + '_0000.nii.gz'))
            shutil.copy(join(basedir, case + '_T2w.nii.gz'), join(imagestr, case + '_0001.nii.gz'))
            shutil.copy(join(basedir, case + '_FLAIR.nii.gz'), join(imagestr, case + '_0002.nii.gz'))
            count+=1


    for patient in os.listdir(val_dir):
        if not patient.startswith('.DS'):
            in_dir = join(val_dir, patient, patient)
            shutil.copy(join(in_dir + '_T1w.nii.gz'), join(imagests, patient + '_0000.nii.gz'))
            shutil.copy(join(in_dir + '_T2w.nii.gz'), join(imagests, patient + '_0001.nii.gz'))
            shutil.copy(join(in_dir + '_FLAIR.nii.gz'), join(imagests, patient + '_0002.nii.gz'))



    generate_dataset_json(out_base, {"0": "T1", "1": "T2", "2": "Flair"},
                          labels={
                              "background": 0,
                              "EPVS": 1},
                          num_training_cases=count, file_ending='.nii.gz',
                          dataset_name=task_name, reference='epvs_v3',
                          release='release',
                          overwrite_image_reader_writer='SimpleITKIO',
                          description="brats",
                          license='')
# def create_brats_1_split(labelsTr_folder: str, seed: int = 1234) -> List[dict[str, List]]:
#     nii_files = nifti_files(labelsTr_folder, join=False)
#     patients = np.unique([i[:len('BraTS-GLI-00000')] for i in nii_files])
#     rs = np.random.RandomState(seed)
#     rs.shuffle(patients)
#     splits = []
#     for fold in range(5):
#         val_patients = patients[fold::5]
#         train_patients = [i for i in patients if i not in val_patients]
#         val_cases = [i[:-7] for i in nii_files for j in val_patients if i.startswith(j)]
#         train_cases = [i[:-7] for i in nii_files for j in train_patients if i.startswith(j)]
#         splits.append({'train': train_cases, 'val': val_cases})
#     return splits


inpath = '/omics/groups/OE0441/E132-Projekte/Projects/2024_Ulrich_epvs/'
taskname = 'epvs_v3'
convert_epvs(taskname, inpath, 418)
