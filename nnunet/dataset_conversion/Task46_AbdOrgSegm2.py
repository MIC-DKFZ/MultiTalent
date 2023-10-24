import numpy as np
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import *
import dicom2nifti
from multiprocessing import Pool
import os
from subprocess import check_output
from nnunet.paths import nnUNet_raw_data
import shutil
# from batchviewer import view_batch


def load_npy_nifti(infile):
    return sitk.GetArrayFromImage(sitk.ReadImage(infile))




def align_img(base, name):
    lab_itk = sitk.ReadImage(join(base, "labelsTr", name + ".nii.gz"))
    img_itk = sitk.ReadImage(join(base, "imagesTr", name + "_0000.nii.gz"))
    img_itk.SetDirection(lab_itk.GetDirection())
    img_itk.SetOrigin(lab_itk.GetOrigin())
    sitk.WriteImage(img_itk, join(base, "imagesTr", name + "_0000.nii.gz"))





def check_alignment(base, name):
    img = load_npy_nifti(join(base, "imagesTr", name + "_0000.nii.gz")).astype(np.float32)
    img[img < -100] = -100
    img[img > 200] = 200
    lab = load_npy_nifti(join(base, "labelsTr", name + ".nii.gz")).astype(np.float32)
    overlay = np.copy(img)
    overlay[lab != 0] -= np.percentile(img, 97)
    print(name)
    # view_batch(img, lab, overlay)




def convert_segmentation(file, output_file, mapping):
    seg = sitk.ReadImage(file)
    seg_npy = sitk.GetArrayFromImage(seg)
    seg_new = np.zeros_like(seg_npy)
    for source_label, target_label in mapping.items():
        seg_new[seg_npy == source_label] = target_label
    seg_new = sitk.GetImageFromArray(seg_new)
    seg_new.CopyInformation(seg)
    sitk.WriteImage(seg_new, output_file)


if __name__ == "__main__":

    task_id = 46
    task_name = "AbdOrgSegm2"


    foldername = "Task%03.0d_%s" % (task_id, task_name)

    out_base = join(nnUNet_raw_data, foldername)
    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")

    base_out = imagestr



    #raw pan ct data
    base = "/home/constantin/Desktop/manifest-1599750808610/Pancreas-CT"
    # BTCV raw data in nnUNet folder structure
    btcv_base = '/home/constantin/E132-Rohdaten/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task017_AbdominalOrganSegmentation'
    #target folder
    #downloaded label raw data
    labels_base = "/home/constantin/Desktop/pancreas/labels"
    #
    maybe_mkdir_p(base_out)
    args = []
    for case in subdirs(base, join=False):
        cur = join(base, case)
        outfile = join(base_out, case + ".nii.gz")
        for t1 in subdirs(cur, join=False):
            curr = join(cur, t1)
            for t2 in subdirs(curr, join=False):
                currr = join(curr, t2)
                if not isfile(outfile):
                    args.append([currr, outfile])


    p = Pool(4)
    p.starmap(dicom2nifti.dicom_series_to_nifti, args)
    p.close()
    p.join()


    # rename to _0000.nii.gz
    for f in [i for i in subfiles(base_out, join=False) if not i.endswith("_0000.nii.gz")]:
        os.rename(join(base_out, f), join(base_out, f[:-7] + "_0000.nii.gz"))

    label_mapping = {
        0: 'background',
        1: "spleen",
        3: "left kidney",
        4: "gallbladder",
        5: "esophagus",
        6: "liver",
        7: "stomach",
        11: "pancreas",
        14: "duodenum"
    }

    label_mapping_new = {i: v for i, (k, v) in enumerate(label_mapping.items())}

    organ_label_map_old = {v: k for k, v in label_mapping.items()}
    organ_label_map_new = {v: k for k, v in label_mapping_new.items()}

    change_labels = {organ_label_map_old[k]:organ_label_map_new[k] for k in organ_label_map_new.keys()}


    current_labels = join(labels_base, "label_tciapancreasct_multiorgan/label_tcia_multiorgan")
    labels_out = labelstr
    maybe_mkdir_p(labels_out)
    # sort out the pancreas cases
    pancreas_cases = subfiles(base_out, prefix="PANCREAS", join=False)
    for p in pancreas_cases:
        idx = int(p[9:13])
        # check if label is present

        if isfile(join(current_labels, "label%04.0d.nii.gz" % idx)):
            convert_segmentation(join(current_labels, "label%04.0d.nii.gz" % idx), join(labels_out, p[:-12] + ".nii.gz"), change_labels)

    # now BTCV
    #copy BTCV images in base_out - also test images (why did they also label the test images... :( )
    for file in os.listdir(join(btcv_base, 'imagesTr')):
        if file.endswith('nii.gz'):
            shutil.copyfile(join(btcv_base, 'imagesTr',  file), join(base_out, file))

    for file in os.listdir(join(btcv_base, 'imagesTs')):
        if file.endswith('nii.gz'):
            shutil.copyfile(join(btcv_base,'imagesTs', file), join(base_out, file))

    current_labels = join(labels_base, "label_btcv_multiorgan")
    bcv_cases = subfiles(base_out, prefix="img", join=False)
    for p in bcv_cases:
        idx = int(p[3:7])
        # check if label is present
        if isfile(join(current_labels, "label%04.0d.nii.gz" % idx)):
            convert_segmentation(join(current_labels, "label%04.0d.nii.gz" % idx), join(labels_out, p[:-12] + ".nii.gz"), change_labels)

    # now remove image files with no labels
    labels = subfiles(labels_out, join=False)
    for f in subfiles(base_out, join=False):
        if f[:-12] + ".nii.gz" not in labels:
            print(f)
            os.remove(join(base_out, f))

    json_dict = {}
    json_dict['name'] = "AbdOrgSegm2"
    json_dict['description'] = "https://zenodo.org/record/1169361#.XR4DNp9fjRY"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "https://zenodo.org/record/1169361#.XR4DNp9fjRY"
    json_dict['licence'] = "https://zenodo.org/record/1169361#.XR4DNp9fjRY"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "CT",
    }
    json_dict['labels'] = {str(k): v for k, v in label_mapping_new.items()}
    json_dict['numTraining'] = len(labels)
    json_dict['numTest'] = 0
    json_dict['training'] = [{'image': "./imagesTr/%s" % i , "label": "./labelsTr/%s" % i} for i in labels]
    json_dict['test'] = []

    with open(os.path.join(out_base, "dataset.json"), 'w') as f:
        json.dump(json_dict, f, indent=4, sort_keys=True)



for file in os.listdir(labelstr):
    if file.startswith('PAN'):
        print(file)
        align_img(out_base, file[:-7])
        # check_alignment(out_base, file[:-7])
