#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import shutil
from typing import Tuple
from warnings import warn
import SimpleITK as sitk
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from multiprocessing import Pool
from batchgenerators.augmentations.utils import resize_segmentation

from nnunet.paths import nnUNet_raw_data, preprocessing_output_dir, network_training_output_dir
from nnunet.dataset_conversion.utils import generate_dataset_json
from nnunet.utilities.image_reorientation import reorient_all_images_in_folder_to_ras
import nibabel as nib
from nnunet.utilities.task_name_id_conversion import convert_task_name_to_id

'''
only public datasets. 
based on original nnU-Net namespaces
added task18
'''

MultiTalent_task_ids = [
    'Task003_Liver',
    'Task006_Lung',
    'Task007_Pancreas',
    'Task008_HepaticVessel',
    'Task009_Spleen',
    'Task010_Colon',
    'Task017_AbdominalOrganSegmentation',
    'Task046_AbdOrgSegm2',
    'Task051_StructSeg2019_Task3_Thoracic_OAR',
    'Task055_SegTHOR',
    'Task062_NIHPancreas',
    'Task064_KiTS_labelsFixed',
    'Task018_PelvicOrganSegmentation'
]

MultiTalent_task_label_maps = {
    'Task003_Liver': ((1, 2), (1, 2)),
    'Task006_Lung': ((1,), (3,)),
    'Task007_Pancreas': ((1, 2), (4, 5)),
    'Task008_HepaticVessel': ((1, 2), (6, 7)),
    'Task009_Spleen': ((1,), (8,)),
    'Task010_Colon': ((1,), (9,)),
    'Task017_AbdominalOrganSegmentation': (
        (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), (10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22)),
    'Task046_AbdOrgSegm2': ((1, 2, 3, 4, 5, 6, 7, 8), (23, 24, 25, 26, 27, 28, 29, 30)),
    'Task051_StructSeg2019_Task3_Thoracic_OAR': ((1, 2, 3, 4, 5, 6), (31, 32, 33, 34, 35, 36)),
    'Task055_SegTHOR': ((1, 2, 3, 4), (37, 38, 39, 40)),
    'Task062_NIHPancreas': ((1,), (41,)),
    'Task064_KiTS_labelsFixed': ((1, 2), (42, 43)),
    'Task018_PelvicOrganSegmentation': ((1, 2, 3, 4), (44, 45, 46, 47)),
}

MultiTalent_labels = {
    1: '03_liver_wo_cancer',
    2: '03_liver_tumor',
    3: '06_lung_nodule',
    4: '07_pancreas_wo_cancer',
    5: '07_pancreas_cancer',
    6: '08_hepatic_vessel',
    7: '08_liver_cancer',
    8: '09_spleen',
    9: '10_colon_cancer',
    10: '17_spleen',
    11: '17_right_kidney',
    12: '17_left_kidney',
    13: '17_gallbladder',
    14: '17_esophagus',
    15: '17_liver_whole',
    16: '17_stomach',
    17: '17_aorta',
    18: '17_inf_vena_cava',
    19: '17_port_and_splen_vein',
    20: '17_pancreas_whole',
    21: '17_right_adrenal_gland',
    22: '17_left_adrenal_gland',
    23: '46_spleen',
    24: '46_left_kidney',
    25: '46_gallbladder',
    26: '46_esophagus',
    27: '46_liver',
    28: '46_stomach',
    29: '46_pancreas',
    30: '46_duodenum',
    31: '51_left_lung',
    32: '51_right_lung',
    33: '51_heart',
    34: '51_esophagus',
    35: '51_bronchies',
    36: '51_spinal_cord_nerve_thingy',
    37: '55_esophagus',
    38: '55_heart',
    39: '55_trachea',
    40: '55_aorta',
    41: '62_pancreas',
    42: '64_both_kidneys_wo_tumor',
    43: '64_kidney_tumor',
    44: '18_bladder',
    45: '18_uterus',
    46: '18_rectum',
    47: '18_small_bowel',
}

MultiTalent_regions = {
    '03_liver': (1, 2),
    '03_cancer': (2,),
    '06_lungnodule': (3,),
    '07_pancreas': (4, 5),
    '07_pancreas_cancer': (5,),
    '08_vessel': (6,),
    '08_tumor': (7,),
    '09_spleen': (8,),
    '10_colon_cancer': (9,),
    '17_spleen': (10,),
    '17_right_kidney': (11,),
    '17_left_kidney': (12,),
    '17_gallbladder': (13,),
    '17_esophagus': (14,),
    '17_liver': (15,),
    '17_stomach': (16,),
    '17_aorta': (17,),
    '17_inf_vena_cava': (18,),
    '17_port_and_splen_vein': (19,),
    '17_pancreas': (20,),
    '17_right_adrenal_gland': (21,),
    '17_left_adrenal_gland': (22,),
    '46_spleen': (23,),
    '46_left_kidney': (24,),
    '46_gallbladder': (25,),
    '46_esophagus': (26,),
    '46_liver': (27,),
    '46_stomach': (28,),
    '46_pancreas': (29,),
    '46_duodenum': (30,),
    '51_left_lung': (31,),
    '51_right_lung': (32,),
    '51_heart': (33,),
    '51_esophagus': (34,),
    '51_bronchies': (35,),
    '51_spinal_cord_nerve_thingy': (36,),
    '55_esophagus': (37,),
    '55_heart': (38,),
    '55_trachea': (39,),
    '55_aorta': (40,),
    '62_pancreas': (41,),
    '64_both_kidneys': (42, 43),
    '64_kidney_tumor': (43,),
    '18_bladder': (44,),
    '18_uterus': (45,),
    '18_rectum': (46,),
    '18_small_bowel': (47,),
}

MultiTalent_regions_class_order = {
    'Task003_Liver': (1, 2),
    'Task006_Lung': (3,),
    'Task007_Pancreas': (4, 5),
    'Task008_HepaticVessel': (6, 7),
    'Task009_Spleen': (8,),
    'Task010_Colon': (9,),
    'Task017_AbdominalOrganSegmentation': (10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22),
    'Task046_AbdOrgSegm2': (23, 24, 25, 26, 27, 28, 29, 30),
    'Task051_StructSeg2019_Task3_Thoracic_OAR': (31, 32, 33, 34, 35, 36),
    'Task055_SegTHOR': (37, 38, 39, 40),
    'Task062_NIHPancreas': (41,),
    'Task064_KiTS_labelsFixed': (42, 43),
    'Task018_PelvicOrganSegmentation': (44, 45, 46, 47)

}

MultiTalent_region_output_idx_mapping = {j: i for i, j in enumerate(MultiTalent_regions.keys())}

MultiTalent_valid_regions = {
    'Task003_Liver': ('03_liver', '03_cancer'),
    'Task006_Lung': ('06_lungnodule',),
    'Task007_Pancreas': ('07_pancreas', '07_pancreas_cancer'),
    'Task008_HepaticVessel': ('08_vessel', '08_tumor'),
    'Task009_Spleen': ('09_spleen',),
    'Task010_Colon': ('10_colon_cancer',),
    'Task017_AbdominalOrganSegmentation': ('17_spleen', '17_right_kidney', '17_left_kidney',
                                           '17_gallbladder', '17_esophagus', '17_liver', '17_stomach',
                                           '17_aorta', '17_inf_vena_cava', '17_port_and_splen_vein', '17_pancreas',
                                           '17_right_adrenal_gland', '17_left_adrenal_gland'),
    'Task046_AbdOrgSegm2': ('46_spleen', '46_left_kidney', '46_gallbladder', '46_esophagus',
                            '46_liver', '46_stomach', '46_pancreas', '46_duodenum'),
    'Task051_StructSeg2019_Task3_Thoracic_OAR': (
        '51_left_lung', '51_right_lung', '51_heart', '51_esophagus', '51_bronchies',
        '51_spinal_cord_nerve_thingy'),
    'Task055_SegTHOR': ('55_esophagus', '55_heart', '55_trachea', '55_aorta'),
    'Task062_NIHPancreas': ('62_pancreas',),
    'Task064_KiTS_labelsFixed': ('64_both_kidneys', '64_kidney_tumor'),
    'Task018_PelvicOrganSegmentation': ('18_bladder', '18_uterus', '18_rectum', '18_small_bowel'),
}


def sanity_checks():
    for t in MultiTalent_valid_regions.keys():
        regions = MultiTalent_valid_regions[t]
        labels = list(np.unique([i for r in regions for i in MultiTalent_regions[r]]))
        assert len(labels) == len(MultiTalent_task_label_maps[t][1])
        assert all([i in MultiTalent_task_label_maps[t][1] for i in labels])

def copy_and_convert_segmentation_nifti(in_file: str, out_file: str, labels_in: Tuple,
                                        labels_out: Tuple[int, ...],
                                        sanity_check: bool = True) -> None:

    img = nib.load(in_file)
    img_npy = np.ascontiguousarray(img.get_fdata())
    seg_new = copy_and_convert_segmentation(img_npy, labels_in, labels_out, sanity_check, in_file)

    img_corr = nib.Nifti1Image(seg_new, img.affine, img.header)
    nib.save(img_corr, out_file)


def copy_and_convert_segmentation(segmentation: np.ndarray, labels_in: Tuple, labels_out: Tuple[int, ...],
                                  sanity_check: bool = True, in_file: str = None) -> np.ndarray:
    """
    labels_in is a tuple. Its entries can be int or tuple. If tuple, then several labels will be mapped to the
    same output label. For example
    labels_in = (1, 2, (3, 4), 3)
    labels_out = (4, 5, 6, 7)

    then 1 will be mapped to 4, 2 -> 5, 3 and 4 will be mapped to 6 and 3-> 7

    labels_out must be a list of int and cannot contain tuples

    sanity_check is True then we check that all unique labels in the segmentation are represented in labels_in.
    """
    assert len(labels_in) == len(labels_out)
    # print(in_file)
    uniques = np.unique(segmentation)
    uniques = uniques[uniques > 1e-20]
    if sanity_check:
        flat_labels_in = []
        for i in labels_in:
            if isinstance(i, int):
                flat_labels_in.append(i)
            else:
                flat_labels_in += list(i)
        flat_labels_in = np.unique(flat_labels_in)
        for u in uniques:
            if u not in flat_labels_in:
                #print('uniques', uniques)
                raise RuntimeError('unexpected labl in image: %d, expected labels are %s',
                                   (u, str(flat_labels_in)))

    seg_new = np.zeros_like(segmentation, dtype=np.uint8)
    for i, o in zip(labels_in, labels_out):
        if not hasattr(i, "__len__"):
            i = (i,)
        for ii in i:
            if ii in uniques:
                if not isinstance(ii, int):
                    print(labels_in)
                    raise AssertionError
                if not isinstance(o, int):
                    print(labels_out)
                    raise AssertionError
                seg_new[segmentation == ii] = o

    return seg_new

if __name__ == '__main__':

    task_name = "Task100_MultiTalent"

    target_base = join(nnUNet_raw_data, task_name)
    target_imagesTr = join(target_base, "imagesTr")
    target_imagesVal = join(target_base, "imagesVal")
    target_labelsTr = join(target_base, "labelsTr")
    target_imagesTs = join(target_base, "imagesTs")
    target_labelsVal = join(target_base, "labelsVal")
    target_labelsTs = join(target_base, "labelsTs")

    maybe_mkdir_p(target_imagesTr)
    maybe_mkdir_p(target_imagesVal)
    maybe_mkdir_p(target_labelsTr)
    maybe_mkdir_p(target_imagesTs)
    maybe_mkdir_p(target_labelsVal)
    maybe_mkdir_p(target_labelsTs)

    cases_have_labels_tr = {}
    cases_have_labels_val = {}
    cases_have_labels_ts = {}
    cases_have_regions_tr = {}
    cases_have_regions_val = {}
    cases_have_regions_ts = {}
    res = []

    #we had some issues with these images, not correct registered?!
    # shape_issues = ['062_pancreas_0045.nii.gz', '062_pancreas_0032.nii.gz', '062_pancreas_0027.nii.gz', '062_pancreas_0007.nii.gz']
    shape_issues = []
    p = Pool(8)
    for t in MultiTalent_task_label_maps.keys():
        print(t)
        task_id = t[4:7]
        expected_source_dir = join(nnUNet_raw_data, t)
        if not isdir(expected_source_dir):
            raise RuntimeError('missing task: %s' % (t))
        imagesTr = join(expected_source_dir, 'imagesTr')
        imagesVal = join(expected_source_dir, 'imagesVal')
        imagesTs = join(expected_source_dir, 'imagesTs')

        labelsTr = join(expected_source_dir, 'labelsTr')
        labelsTs = join(expected_source_dir, 'labelsTs')
        labelsVal = join(expected_source_dir, 'labelsVal')

        # interleaving data and seg copies is important
        indir = imagesTr
        outdir = target_imagesTr
        images = subfiles(indir, suffix='nii.gz', join=False)
        for i in images:
            target_fname = join(outdir, task_id + '_' + i)
            if task_id + '_' + i[:-12] + '.nii.gz' in shape_issues:
                continue
            else:
                if not isfile(target_fname):
                    shutil.copy(join(indir, i), target_fname)

        indir = labelsTr
        outdir = target_labelsTr
        images = subfiles(indir, suffix='nii.gz', join=False)
        for i in images:
            target_fname = join(outdir, task_id + '_' + i)
            if task_id + '_' + i in shape_issues:
                continue
            else:
                if not isfile(target_fname):
                    res.append(p.starmap_async(copy_and_convert_segmentation_nifti,
                    ((join(indir, i),target_fname,MultiTalent_task_label_maps[t][0],MultiTalent_task_label_maps[t][1],),)))

            cases_have_labels_tr[task_id + '_' + i] = MultiTalent_task_label_maps[t][1]
            cases_have_regions_tr[task_id + '_' + i] = MultiTalent_valid_regions[t]

        if isdir(imagesVal):
            indir = imagesVal
            outdir = target_imagesVal
            images = subfiles(indir, suffix='nii.gz', join=False)
            for i in images:
                target_fname = join(outdir, task_id + '_' + i)
                if not isfile(target_fname):
                    shutil.copy(join(indir, i), target_fname)

        if isdir(labelsVal):
            indir = labelsVal
            outdir = target_labelsVal
            images = subfiles(indir, suffix='nii.gz', join=False)
            for i in images:
                target_fname = join(outdir, task_id + '_' + i)
                res.append(p.starmap_async(copy_and_convert_segmentation_nifti,
                ((join(indir, i), target_fname, MultiTalent_task_label_maps[t][0], MultiTalent_task_label_maps[t][1],),)))

                cases_have_labels_val[task_id + '_' + i] = MultiTalent_task_label_maps[t][1]
                cases_have_regions_val[task_id + '_' + i] = MultiTalent_valid_regions[t]

        if isdir(imagesTs):
            indir = imagesTs
            outdir = target_imagesTs
            images = subfiles(indir, suffix='nii.gz', join=False)
            for i in images:
                target_fname = join(outdir, task_id + '_' + i)
                if not isfile(target_fname):
                    shutil.copy(join(indir, i), target_fname)

        if isdir(labelsTs):
            indir = labelsTs
            outdir = target_labelsTs
            images = subfiles(indir, suffix='nii.gz', join=False)
            for i in images:
                target_fname = join(outdir, task_id + '_' + i)
                if not isfile(target_fname):
                    res.append(p.starmap_async(copy_and_convert_segmentation_nifti,
                    ((join(indir, i),target_fname,MultiTalent_task_label_maps[t][0],MultiTalent_task_label_maps[t][1],),)))

                cases_have_labels_ts[task_id + '_' + i] = MultiTalent_task_label_maps[t][1]
                cases_have_regions_ts[task_id + '_' + i] = MultiTalent_valid_regions[t]

    _ = [i.get() for i in res]
    print('done')


    generate_dataset_json(join(target_base, 'dataset.json'), target_imagesTr, target_imagesTs, ("CT",),
                          MultiTalent_labels,
                          task_name)
    cases_labels_and_regions = (cases_have_labels_tr, cases_have_labels_val, cases_have_labels_ts,
                                cases_have_regions_tr, cases_have_regions_val, cases_have_regions_ts)
    save_pickle(cases_labels_and_regions, join(target_base, 'cases_have_regions_labels.pkl'))







