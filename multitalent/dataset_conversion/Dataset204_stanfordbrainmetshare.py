import os
from batchgenerators.utilities.file_and_folder_operations import *
import SimpleITK as sitk
import imageio
import numpy as np
from multitalent.dataset_conversion.generate_dataset_json import generate_dataset_json


def threed_mat_to_basic_nifti(arr: np.ndarray, save_p: str) -> None:
    # Creates the directory to write to, Meta Data non-existent therefore left empty at default values
    os.makedirs(os.path.dirname(save_p), exist_ok=True)

    nfti: sitk.Image = sitk.GetImageFromArray(arr)
    # nfti.SetOrigin(origin=[0, 0, 0])
    # nfti.SetDirection(direction=[1., 0., 0., 0., 1., 0., 0., 0., 1.])
    nfti.SetSpacing([0.94, 0.94, 1.])
    sitk.WriteImage(nfti, save_p)


def main():
    # There are 2 variables with 2 settings to evaluate this Dataset:
    # 1: Spin2Spin SubImage or Spin2Grad Subimage
    # 2: Original GT or Selflabeled GT

    base_p ='/home/constantin/cluster-data/multitalent_floy/stanford_brain_mets_train_with_labels'
    out_p = '/home/constantin/cluster-data/multitalent_floy/Dataset204_stanfordbrainmetshare'
    #### Train stuff
    train_n = "mets_stanford_releaseMask_train"
    train_im_out_n = os.path.join(out_p, "imagesTr")
    train_lbl_out_n = os.path.join(out_p, "labelsTr")
    test_im_out_n = os.path.join(out_p, "imagesTs")
    test_lbl_out_n = os.path.join(out_p, "labelsTs")

    maybe_mkdir_p(train_im_out_n)
    maybe_mkdir_p(train_lbl_out_n)
    maybe_mkdir_p(test_im_out_n)
    maybe_mkdir_p(test_lbl_out_n)

    train_p = os.path.join(base_p, train_n)
    subdirs = os.listdir(train_p)
    all_patiens = []
    for p in subdirs:
        if p.startswith('Mets'):
            all_patiens.append(p)


    rnd = np.random.RandomState(seed=12345)
    all_patiens = np.sort(all_patiens)
    idx_tr = rnd.choice(len(all_patiens), int(len(all_patiens) * 0.8), replace=False)
    casesTr = [all_patiens[i] for i in idx_tr]


    for patient_name in sorted(all_patiens):
        patient_p = os.path.join(train_p, patient_name)
        class_0_p = os.path.join(patient_p, "0")
        class_1_p = os.path.join(patient_p, "1")
        class_2_p = os.path.join(patient_p, "2")
        class_3_p = os.path.join(patient_p, "3")
        seg_p = os.path.join(patient_p, "seg")

        # '0' contains T1 gradient-echo post images
        # '1' contains T1 spin-echo pre images
        # '2' contains T1 spin-echo post images
        # '3' contains T2 FLAIR post images
        # 'seg' contains a binary mask of the segmented metastases (0, 255)
        paths = [class_0_p, class_1_p, class_2_p, class_3_p, seg_p]
        dims = [0, 1, 2, 3, "seg"]
        for cur_patient_dp, cur_dim in zip(paths, dims):
            all_slices = []
            for im_slice in sorted(os.listdir(class_0_p)):
                im = imageio.imread(os.path.join(cur_patient_dp, im_slice))
                all_slices.append(im)
            threed_im = np.stack(all_slices, axis=0)
            # threed_im_tp = np.transpose(threed_im, axes=[1, 2, 0])
            if patient_name in casesTr:
                if type(cur_dim) == int:
                    out_name = patient_name + f"_{cur_dim:04}.nii.gz"
                    out_path = os.path.join(train_im_out_n, out_name)
                else:
                    out_name = patient_name + ".nii.gz"
                    out_path = os.path.join(train_lbl_out_n, out_name)
                    threed_im[threed_im == 255] = 1
            else:
                if type(cur_dim) == int:
                    out_name = patient_name + f"_{cur_dim:04}.nii.gz"
                    out_path = os.path.join(test_im_out_n, out_name)
                else:
                    out_name = patient_name + ".nii.gz"
                    out_path = os.path.join(test_lbl_out_n, out_name)
                    threed_im[threed_im == 255] = 1

            threed_mat_to_basic_nifti(threed_im, out_path)

    del train_im_out_n, train_lbl_out_n, train_n




if __name__ == '__main__':
    #main()
    output_folder = '/home/constantin/cluster-data/multitalent_floy/Dataset204_stanfordbrainmetshare'

    labels = {"background" : 0, "enhancing" : 1}
    channel_names = {"T1 gradient-echo post" : 0, "T1 spin-echo pre" : 1, "T1 spin-echo post" : 2, "T2 FLAIR post images" : 3}
    num_training_cases = 84
    dataset_name = 'stanfordbrainmetshare'

    generate_dataset_json(output_folder, channel_names, labels, num_training_cases, 'nii.gz', dataset_name)
