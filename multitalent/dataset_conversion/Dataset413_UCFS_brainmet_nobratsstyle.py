from batchgenerators.utilities.file_and_folder_operations import *
import shutil
import numpy as np
import SimpleITK as sitk
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
        if patient.startswith('100'):
            if not isfile(join(basedir,patient,patient +'_BraTS-seg.nii.gz')):
                count +=1
                t1_path = join(basedir,patient,patient +'_T1pre.nii.gz')
                t1c_path = join(basedir,patient,patient +'_T1post.nii.gz')
                t2_path = join(basedir,patient,patient +'_T2Synth.nii.gz')
                t2f_path = join(basedir,patient,patient +'_FLAIR.nii.gz')
                seg_path = join(basedir,patient,patient +'_seg.nii.gz')
                t1_img = sitk.ReadImage(t1_path)
                t1c_img = sitk.ReadImage(t1c_path)
                t2_img = sitk.ReadImage(t2_path)
                t2f_img = sitk.ReadImage(t2f_path)
                seg_img = sitk.ReadImage(seg_path)

                t1_img.CopyInformation(seg_img)
                t1c_img.CopyInformation(seg_img)
                t2_img.CopyInformation(seg_img)
                t2f_img.CopyInformation(seg_img)


                sitk.WriteImage(t1_img, join(imagestr, patient + '_0000.nii.gz'))
                sitk.WriteImage(t1c_img, join(imagestr, patient + '_0001.nii.gz'))
                sitk.WriteImage(t2_img, join(imagestr, patient + '_0002.nii.gz'))
                sitk.WriteImage(t2f_img, join(imagestr, patient + '_0003.nii.gz'))
                sitk.WriteImage(seg_img, join(labelstr, patient + '.nii.gz'))




    generate_dataset_json(out_base, {"0": "T1", "1": "T1ce", "2": "T2", "3": "Flair"},
                          labels={
                              "background": 0,
                              "ET": 1},
                          num_training_cases=count, file_ending='.nii.gz',
                          dataset_name=task_name, reference='UCFS_brainmet_forbrats',
                          release='release',
                          overwrite_image_reader_writer='SimpleITKIO',
                          description="brats",
                          license='')

def create_ucfs_split(labelsTr_folder: str, seed: int = 1234) -> List[dict[str, List]]:
    nii_files = nifti_files(labelsTr_folder, join=False)
    patients = np.unique([i[:len('100195')] for i in nii_files])
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


inpath = '/omics/groups/OE0441/E132-Projekte/Projects/2024_Ulrich_beyond_brats/UCSF_BrainMetastases_v1.3/UCSF_BrainMetastases_TRAIN'
taskname = 'UCFS_brainmet_nobratsstyle'
# convert_brats(taskname, inpath, 413)
dataset_name = 'Dataset413_' + taskname
preprocessed_folder = join(nnUNet_preprocessed, dataset_name)
maybe_mkdir_p(preprocessed_folder)
split = create_ucfs_split(join(nnUNet_raw, dataset_name, 'labelsTr'))
save_json(split, join(preprocessed_folder, 'splits_final.json'), sort_keys=False)