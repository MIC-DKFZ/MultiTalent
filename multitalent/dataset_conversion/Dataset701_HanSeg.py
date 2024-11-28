from batchgenerators.utilities.file_and_folder_operations import *
import shutil
from multitalent.dataset_conversion.generate_dataset_json import generate_dataset_json
from multitalent.paths import nnUNet_raw
import SimpleITK as sitk
import numpy as np
import os
import sys
import math
import fnmatch
import numpy as np
import SimpleITK as sitk
from importlib import reload
import matplotlib.pyplot as plt

from scipy import ndimage
from scipy.stats import norm


def paddingSitkImage(im_sitk, target_shape, pad_value, output_format="sitk", verbose=False):
    '''The function is zero-padding the image for a given size
    Parameters:
        - im_sitk: input image as an SitkImage
        - target_shape: list or tuple, new shape in the order of z(transversal plane), y, x
        - output_format: string, type of output: "sitk", "np"
        - verbose: bool, set it True if you want to print information about the padding
        - im_sitk_pad: SitkImage, the output image
    '''
    # Calculate padding on both sides of the image
    im_nparr = sitk.GetArrayFromImage(im_sitk)
    bz, by, bx = ((np.subtract(target_shape, im_nparr.shape)) / 2)
    if bz < 0 or by < 0 or bx < 0:
        print(bz, by, bx)
        print("ERROR: One of the target dimensions is smaller than the original image, consider cropping")
        return
    bzh, byh, bxh = np.ceil((bz, by, bx)).astype(int)
    bzl, byl, bxl = np.floor((bz, by, bx)).astype(int)

    # Padding:
    im_nparr_pad = np.pad(im_nparr, ((bzh, bzl), (byh, byl), (bxh, bxl)), 'constant', constant_values=pad_value)
    if verbose:
        print("PADDING:")
        print("Input Shape: ", im_nparr.shape)
        print("Padding - z: {:3d} + {:3d}".format(bzl, bzh))
        print("        - y: {:3d} + {:3d}".format(byl, byh))
        print("        - x: {:3d} + {:3d}".format(bxl, bxh))
        print("New Shape:   ", im_nparr_pad.shape)
        print()

    if output_format == "sitk":
        # Retransform nparray to SitkImage
        im_sitk_pad = sitk.GetImageFromArray(im_nparr_pad)
        im_sitk_pad.SetOrigin(im_sitk.GetOrigin())
        im_sitk_pad.SetSpacing(im_sitk.GetSpacing())
        im_sitk_pad.SetDirection(im_sitk.GetDirection())
        return im_sitk_pad
    elif output_format == "np":
        return im_nparr_pad
    else:
        print("ERROR: Unknown output format")
        return


def respaceSitkImage(image, new_spacing=[0.3125, 0.3125, 3.0], ignore_zratio=False, interpolation="nearest",
                     verbose=False):
    '''The function respaces the image with a given spacing and interpolation
    Parameters:
        - image: input image as an SitkImage
        - new_spacing: list, new spacing in the order of x, y, z(transversal plane)
        - ignore_zratio: bool, True if you dont want to get interpolation error in the z-axis
                         due to the negligibly small deviation in the z spacing
        - interpolation: string, type of interpolation: "bspline", "linear", "nearest"
        - verbose: bool, True if you want to print information about the respacing steps
    '''

    # Calculating the ratio of the old and the new spacing:
    orig_spacing = image.GetSpacing()
    if ignore_zratio:
        overwrite_zspacing = new_spacing[2]
        new_spacing[2] = image.GetSpacing()[2]
    spacing_ratio = np.divide(orig_spacing, new_spacing)
    if verbose:
        print("SPACING RATIO:")
        print("Original Spacing: {:7.5f}, {:5.5f}, {:10.8f}".format(orig_spacing[0], orig_spacing[1], orig_spacing[2]))
        print("New Spacing:      {:6.5f}, {:5.5f}, {:10.8f}".format(new_spacing[0], new_spacing[1], new_spacing[2]))
        print(
            "Spacing Ratio:    {:6.5f}, {:5.5f}, {:10.8f}".format(spacing_ratio[0], spacing_ratio[1], spacing_ratio[2]))
        print()

    # Calculating the new image size based on the spacing ratio
    orig_size = image.GetSize()
    new_size = orig_size * spacing_ratio
    for i in range(len(new_spacing)):
        if (new_size[i] - int(new_size[i])) < 0.5:
            new_size[i] = math.floor(new_size[i]) - 1
        else:
            new_size[i] = math.ceil(new_size[i]) - 1
    new_size = [int(s) for s in new_size]
    if verbose:
        print("IMAGE SIZES:")
        print("Original Size:    {:3d}, {:3d}, {:2d}".format(orig_size[0], orig_size[1], orig_size[2]))
        print("New Size:         {:3d}, {:3d}, {:2d}".format(new_size[0], new_size[1], new_size[2]))
        print()

    # Setting up the SITK Resampler and execute it
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(new_spacing)
    if interpolation == "bspline":
        resampler.SetInterpolator(sitk.sitkBSpline)
    elif interpolation == "linear":
        resampler.SetInterpolator(sitk.sitkLinear)
    elif interpolation == "nearest":
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        print("ERROR: Unknown interpolation method")
        return

    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampled_image = resampler.Execute(image)
    if ignore_zratio:
        resampled_image.SetSpacing([new_spacing[0], new_spacing[1], overwrite_zspacing])

    if verbose:
        print("FINAL SPACING:")
        print("New Spacing:     ", resampled_image.GetSpacing())
        print()

    return resampled_image


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

def convert_HanSeg(base_dir: str, nnunet_dataset_id: int = 701):
    task_name = "HanSeg"

    foldername = "Dataset%03.0d_%s" % (nnunet_dataset_id, task_name)

    # setting up nnU-Net folders
    out_base = join(nnUNet_raw, foldername)
    imagestr = join(out_base, "imagesTr")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)

    cases = subdirs(base_dir, prefix='case_', join=False)

    for name in cases:
        print(name)
        if not isfile(join(labelstr, name + '.nrrd')):
            case_base_dir = join(base_dir, name)
            ct_img = sitk.ReadImage(join(case_base_dir, name + '_IMG_CT.nrrd'))
            # im_MR = sitk.ReadImage(join(case_base_dir, name + '_IMG_MR_T1.nrrd'))
            #
            #
            # #registration
            # im_MR.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
            # im_MR_res = respaceSitkImage(image=im_MR,
            #                              new_spacing=ct_img.GetSpacing(),
            #                              ignore_zratio=False,
            #                              interpolation="nearest",
            #                              verbose=True)
            #
            # im_MR_respad = paddingSitkImage(im_sitk=im_MR_res, target_shape=ct_img.GetSize()[::-1],
            #                                 pad_value=im_MR.GetPixel((0, 0, 0)), output_format="sitk", verbose=True)
            # im_MR_respad.SetOrigin(ct_img.GetOrigin())
            #
            # fixedImage = ct_img
            # movingImage = im_MR_respad
            #
            # parameterMap = sitk.GetDefaultParameterMap('rigid')
            # sitk.PrintParameterMap(parameterMap)
            # parameterMap['AutomaticTransformInitialization'] = ['true']
            # parameterMap['AutomaticTransformInitializationMethod'] = ["CenterOfGravity"]
            #
            # # parameterMap['MaximumNumberOfIterations'] = ['512']
            # # parameterMap['MaximumNumberOfSamplingAttempts'] = ['124']
            # # parameterMap['NumberOfSamplesForExactGradient'] = ['64000']
            # # parameterMap['NumberOfSpatialSamples'] = ['512000']
            # sitk.PrintParameterMap(parameterMap)
            #
            # elastixImageFilter = sitk.ElastixImageFilter()
            # elastixImageFilter.SetFixedImage(fixedImage)
            # elastixImageFilter.SetMovingImage(movingImage)
            # elastixImageFilter.SetParameterMap(parameterMap)
            #
            # elastixImageFilter.Execute()
            #
            # resultImage = elastixImageFilter.GetResultImage()
            # transformParameterMap = elastixImageFilter.GetTransformParameterMap()
            # sitk.WriteImage(resultImage, join(imagestr, name + '_0001.nrrd'))
            sitk.WriteImage(ct_img, join(imagestr, name + '_0000.nrrd'))

            ct_arr = sitk.GetArrayFromImage(ct_img)
            label_arr = np.zeros(np.shape(ct_arr), np.int8)
            for key in LABEL_dict.keys():
                if key != 'background' and isfile(join(case_base_dir, name + '_OAR_' + key +'.seg.nrrd')):
                    bin_label = sitk.ReadImage(join(case_base_dir, name + '_OAR_' + key +'.seg.nrrd'))
                    bin_label_ar = sitk.GetArrayFromImage(bin_label)
                    label_arr[bin_label_ar==1] = LABEL_dict[key]

            label_img = sitk.GetImageFromArray(label_arr.astype(np.int8))
            label_img.CopyInformation(ct_img)
            sitk.WriteImage(label_img, join(labelstr, name + '.nrrd'))

        generate_dataset_json(out_base, {0: "CT", 1: "T1"},
                              labels=LABEL_dict,
                              num_training_cases=42, file_ending='.nrrd',
                              dataset_name=task_name, reference='none',
                              release='challenge',
                              description="HanSeg2023/24")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-input_folder', type=str,
                        help="The downloaded and extracted HanSeg dataset (must have case_XXXXX subfolders)")
    parser.add_argument('-d', required=False, type=int, default=220, help='nnU-Net Dataset ID, default: 220')
    args = parser.parse_args()
    hanseg_b = args.input_folder
    convert_HanSeg(hanseg_b, args.d)














