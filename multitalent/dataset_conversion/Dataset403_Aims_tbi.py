from batchgenerators.utilities.file_and_folder_operations import *
import shutil
import numpy as np
from multitalent.dataset_conversion.generate_dataset_json import generate_dataset_json
from multitalent.paths import nnUNet_raw
import SimpleITK as sitk
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


    count = 0
    for patient in os.listdir(basedir):
        if patient.endswith('Lesion.nii.gz'):
            count +=1
            img = sitk.ReadImage(join(basedir, patient))
            arr = sitk.GetArrayFromImage(img)
            # uni = np.unique(arr)
            new_arr = np.zeros(np.shape(arr), dtype=np.uint8)
            new_arr[arr>0.1] = 1
            new_img = sitk.GetImageFromArray(new_arr)
            new_img.CopyInformation(img)
            sitk.WriteImage(new_img, join(labelstr, patient[:-14] +'.nii.gz' ))
            shutil.copy(join(basedir, patient[:-14] + '_T1.nii.gz'), join(imagestr, patient[:-14] + '_0000.nii.gz' ))



    generate_dataset_json(out_base, {0: "T1"},
                          labels={
                              "background": 0,
                              "lesion": 1
                          },
                          num_training_cases=count, file_ending='.nii.gz',
                          dataset_name=task_name, reference='Aims_tbi_challenge',
                          release='release',
                          description="")



inpath = '/home/constantin/E132-Projekte/Projects/2024_Ulrich_Aims_tbi/ChallengeFiles'
taskname = 'Aims_tbi'
convert_ms(taskname, inpath, 403)