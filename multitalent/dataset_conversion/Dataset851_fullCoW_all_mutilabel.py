import os

from batchgenerators.utilities.file_and_folder_operations import *
import shutil
import numpy as np
import SimpleITK as sitk
from multitalent.dataset_conversion.generate_dataset_json import generate_dataset_json
from multitalent.paths import nnUNet_raw
print(nnUNet_raw)

def read_roi_metadata(file_path):
    metadata = {}
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('Size (Voxels):'):
                size = line.strip().split(': ')[1].split()
                metadata['s'] = [int(dim) for dim in size]
            elif line.startswith('Location (Voxels):'):
                location = line.strip().split(': ')[1].split()
                metadata['l'] = [int(coord) for coord in location]
    return metadata


def convert_topcowct(taskname: str, basedir: str, nnunet_dataset_id: int,):
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

    imgs = os.listdir(join(basedir, 'imagesTr',))
    labels = os.listdir(join(basedir, 'labelsTr',))


    for tr in imgs:
        if not isfile(join(imagestr, tr)):
            shutil.copy(join(basedir, 'imagesTr', tr), join(imagestr, tr))


    for l in labels:
        label = sitk.ReadImage(join(basedir, 'labelsTr',l))
        if '_ct_' in l:
            roy_file = join('/home/constantin/E132-Projekte/Projects/2023_Rokuss_TopCoW/nnUNet_raw/TopCoW_ROIs', 'topcow_ct_roi_'+l[-10:-6] + 'txt')
        else:
            roy_file = join('/home/constantin/E132-Projekte/Projects/2023_Rokuss_TopCoW/nnUNet_raw/TopCoW_ROIs', 'topcow_mr_roi_' + l[-10:-6] + 'txt')
        roi = read_roi_metadata(roy_file)


        label_arr = sitk.GetArrayFromImage(label)
        new_arr = np.zeros(label_arr.shape, dtype=label_arr.dtype)
        new_arr += 14
        new_arr[roi['l'][2]: roi['l'][2] + roi['s'][2], roi['l'][1]: roi['l'][1] + roi['s'][1], roi['l'][0]: roi['l'][0] + roi['s'][0]] = label_arr[roi['l'][2]: roi['l'][2] + roi['s'][2], roi['l'][1]: roi['l'][1] + roi['s'][1], roi['l'][0]: roi['l'][0] + roi['s'][0]]
        new_img = sitk.GetImageFromArray(new_arr)
        new_img.CopyInformation(label)
        sitk.WriteImage(new_img, join(labelstr, l))




inpath = '/home/constantin/E132-Projekte/Projects/2023_Rokuss_TopCoW/nnUNet_raw/Dataset861_fullCoW_ct_mr_multilabel'
taskname = 'fullCoW_ct_mr_multilabel'
convert_topcowct(taskname, inpath, 851)