from batchgenerators.utilities.file_and_folder_operations import *
import SimpleITK as sitk
from multitalent.dataset_conversion.generate_dataset_json import generate_dataset_json
import numpy as np


def crop_negative_regions(img_arr, label_arr):
    # Create a mask for regions where all values along a dimension are less than 0
    # For each dimension (depth, height, width), find the bounds where values are >= 0
    depth_non_negative = np.any(img_arr >= 10, axis=(1, 2))
    height_non_negative = np.any(img_arr >= 10, axis=(0, 2))
    width_non_negative = np.any(img_arr >= 10, axis=(0, 1))

    # Get the bounding box based on the non-negative regions
    depth_min, depth_max = np.where(depth_non_negative)[0][[0, -1]]
    height_min, height_max = np.where(height_non_negative)[0][[0, -1]]
    width_min, width_max = np.where(width_non_negative)[0][[0, -1]]

    # Crop the img_arr and label_arr using the calculated bounds
    cropped_img_arr = img_arr[depth_min:depth_max + 1, height_min:height_max + 1, width_min:width_max + 1]
    cropped_label_arr = label_arr[depth_min:depth_max + 1, height_min:height_max + 1, width_min:width_max + 1]

    return cropped_img_arr, cropped_label_arr


LABEL_dict = {
    "background": 0,
    "A_Carotid_L": 1,
    "A_Carotid_R": 2,
    "Arytenoid": 3,
    "Bone_Mandible": 4,
    "Brainstem": 5,
    "BuccalMucosa": 6,
    "Cavity_Oral": 7,
    "Cochlea_L": 8,
    "Cochlea_R": 9,
    "Cricopharyngeus": 10,
    "Esophagus_S": 11,
    "Eye_AL": 12,
    "Eye_AR": 13,
    "Eye_PL": 14,
    "Eye_PR": 15,
    "Glnd_Lacrimal_L": 16,
    "Glnd_Lacrimal_R": 17,
    "Glnd_Submand_L": 18,
    "Glnd_Submand_R": 19,
    "Glnd_Thyroid": 20,
    "Glottis": 21,
    "Larynx_SG": 22,
    "Lips": 23,
    "OpticChiasm": 24,
    "OpticNrv_L": 25,
    "OpticNrv_R": 26,
    "Parotid_L": 27,
    "Parotid_R": 28,
    "Pituitary": 29,
    "SpinalCord": 30,
}


# I already have registered MRI images and label maps -> only need to crop MRI images ans labels to non zeoro regions
in_path = '/dkfz/E132-Projekte/Projects/2024_Ulrich_collection/data_multitalent/Dataset701_HanSeg'
out_path = '/dkfz/E132-Projekte/Projects/2024_Ulrich_collection/floy/Dataset703_HanSeg_MRI'
imagesTr = join(out_path, 'imagesTr')
maybe_mkdir_p(imagesTr)
labelsTr = join(out_path, 'labelsTr')
maybe_mkdir_p(labelsTr)



for c,label in enumerate(os.listdir(join(in_path, 'labelsTr'))):
    print(label)
    label_org = sitk.ReadImage(join(in_path, 'labelsTr', label))
    img_org = sitk.ReadImage(join(in_path, 'imagesTr', label[:-5] + '_0001.nrrd'))
    img_arr = sitk.GetArrayFromImage(img_org)
    label_arr = sitk.GetArrayFromImage(label_org)

    cropped_img, cropped_label = crop_negative_regions(img_arr, label_arr)

    img_cropped = sitk.GetImageFromArray(cropped_img.astype(np.float32))
    label_cropped = sitk.GetImageFromArray(cropped_label.astype(np.uint8))

    img_cropped.SetSpacing(img_org.GetSpacing())
    label_cropped.SetSpacing(img_org.GetSpacing())

    sitk.WriteImage(img_cropped, join(imagesTr, label[:-5] +'_0000.nii.gz'))
    sitk.WriteImage(label_cropped, join(labelsTr, label[:-5] +'.nii.gz'))

generate_dataset_json(out_path, {0: "T1"},
                      labels=LABEL_dict,
                      num_training_cases=c+1, file_ending='.nii.gz',
                      dataset_name='Dataset703_HanSeg_MRI', reference='none',
                      release='challenge',
                      description="HanSeg2023/24")




