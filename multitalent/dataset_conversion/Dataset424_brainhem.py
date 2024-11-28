from batchgenerators.utilities.file_and_folder_operations import *
import shutil
import numpy as np
import SimpleITK as sitk
from multitalent.dataset_conversion.generate_dataset_json import generate_dataset_json
from multitalent.paths import nnUNet_raw
print(nnUNet_raw)



def convert_ms(taskname: str, basedir: str, nnunet_dataset_id: int):
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
    d_path = join(basedir, 'ct_scans')
    gt_path = join(basedir, 'masks')
    for patient in os.listdir(d_path):
        if patient.endswith('.nii'):
            count += 1
            #img = sitk.ReadImage(join(d_path, patient))
            #sitk.WriteImage(img, join(imagestr, patient[:-4] +'.nii.gz'))
            label = sitk.ReadImage(join(gt_path, patient))
            arr =sitk.GetArrayFromImage(label)
            arr[arr>0.1]=1
            label_c = sitk.GetImageFromArray(arr)
            label_c.CopyInformation(label)
            sitk.WriteImage(label_c, join(labelstr, patient[:-4] +'.nii.gz'))


    generate_dataset_json(out_base, {0: "CT"},
                          labels={
                              "background": 0,
                              "stroke": 1,
                          },
                          num_training_cases=count, file_ending='.nii.gz',
                          dataset_name=task_name, reference='instance22',
                          release='release',
                          overwrite_image_reader_writer='SimpleITKIO',
                          description="bleedbrain",
                          license='')



inpath = '/home/c306h/E132-Projekte/Projects/2024_ulrich_mbhseg/computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.3.1'
taskname = 'brainhem'
# pseudo_dir = '/home/constantin/E132-Projekte/Projects/2024_ulrich_mbhseg'
convert_ms(taskname, inpath, 424)
