from batchgenerators.utilities.file_and_folder_operations import *
import shutil
import numpy as np
import SimpleITK as sitk
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
        if patient.startswith('BraTS-MEN-RT'):
            #nan in img
            if not patient.startswith('BraTS-MEN-RT-0173-1'):
                count +=1
                t1c_path = join(basedir,patient,patient +'_t1c.nii.gz')
                seg_path = join(basedir,patient,patient +'_gtv.nii.gz')
                # img = sitk.ReadImage(t1c_path)
                # arr = sitk.GetArrayFromImage(img)
                # if np.max(arr) > 1e5:
                #     print(patient)
                # if np.isnan(np.max(arr)):
                #     print(patient, 'maxnan')
                # if np.isnan(np.min(arr)):
                #     print(patient, 'minnan')
            # print(np.unique(arr))
            shutil.copy(t1c_path, join(imagestr, patient + '_0000.nii.gz'))
            shutil.copy(seg_path, join(labelstr, patient + '.nii.gz'))





    generate_dataset_json(out_base, {"0": "T1c"},
                          labels={
                              "background": 0,
                              "GTV": 1},
                          num_training_cases=count, file_ending='.nii.gz',
                          dataset_name=task_name, reference='brats24_task3_MenRT',
                          release='release',
                          overwrite_image_reader_writer='SimpleITKIO',
                          description="brats",
                          license='')



# inpath = '/omics/groups/OE0441/E132-Projekte/Projects/2024_Ulrich_beyond_brats/BraTS2024-MEN-RT-TrainingData/BraTS-MEN-RT-Train-v2'
inpath = '/home/constantin/E132-Projekte/Projects/2024_Ulrich_beyond_brats/BraTS2024-MEN-RT-TrainingData/BraTS-MEN-RT-Train-v2'
taskname = 'brats24_task3_MenRT'
convert_brats(taskname, inpath, 407)