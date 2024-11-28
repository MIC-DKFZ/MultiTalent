from batchgenerators.utilities.file_and_folder_operations import *
import shutil
import numpy as np
from multitalent.dataset_conversion.generate_dataset_json import generate_dataset_json
from multitalent.paths import nnUNet_raw
import SimpleITK as sitk
print(nnUNet_raw)





def convert_isles(taskname: str, basedir: str, nnunet_dataset_id: int,):
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

    ct_imgbase = join(basedir, 'raw_data')


    count = 0
    for patient in os.listdir(ct_imgbase):
        if patient.startswith('sub-stroke'):
            ctimgpath = join(basedir, 'raw_data', patient, 'ses-01', patient + '_ses-01_ncct.nii.gz')
            ctaimgpath = join(basedir, 'derivatives', patient, 'ses-01', patient + '_ses-01_space-ncct_cta.nii.gz')
            cbfimgpath = join(basedir, 'derivatives', patient, 'ses-01', 'perfusion-maps',  patient + '_ses-01_space-ncct_cbf.nii.gz')
            cbvimgpath = join(basedir, 'derivatives', patient, 'ses-01', 'perfusion-maps',  patient + '_ses-01_space-ncct_cbv.nii.gz')
            mttimgpath = join(basedir, 'derivatives', patient, 'ses-01', 'perfusion-maps',  patient + '_ses-01_space-ncct_mtt.nii.gz')
            tmaximgpath = join(basedir, 'derivatives', patient, 'ses-01', 'perfusion-maps',  patient + '_ses-01_space-ncct_tmax.nii.gz')
            labelpath = join(basedir, 'derivatives', patient, 'ses-02', patient + '_ses-02_lesion-msk.nii.gz')

            shutil.copy(ctimgpath, join(imagestr, patient + '_0000.nii.gz'))
            shutil.copy(ctaimgpath, join(imagestr, patient + '_0001.nii.gz'))
            shutil.copy(cbfimgpath, join(imagestr, patient + '_0002.nii.gz'))
            shutil.copy(cbvimgpath, join(imagestr, patient + '_0003.nii.gz'))
            shutil.copy(mttimgpath, join(imagestr, patient + '_0004.nii.gz'))
            shutil.copy(tmaximgpath, join(imagestr, patient + '_0005.nii.gz'))
            shutil.copy(labelpath, join(labelstr, patient + '.nii.gz'))
            count+=1
            print(count)






    generate_dataset_json(out_base, {0: "CT", 1: "CT", 2: 'CTP_cbf', 3: 'CTP_cbv', 4: 'CTP_mtt', 2: 'CTP_tmax', },
                          labels={
                              "background": 0,
                              "lesion": 1
                          },
                          num_training_cases=count, file_ending='.nii.gz',
                          dataset_name=task_name, reference='isles 24 challenge',
                          release='release',
                          description="")



inpath = '/home/constantin/E132-Projekte/Projects/2024_Zenk_ISLES_challenge/isles24_batch_1'
taskname = 'isles24'
convert_isles(taskname, inpath, 404)