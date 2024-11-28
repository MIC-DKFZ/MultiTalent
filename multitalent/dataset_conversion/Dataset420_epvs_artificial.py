from batchgenerators.utilities.file_and_folder_operations import *
import shutil
import numpy as np
import SimpleITK as sitk
from multitalent.dataset_conversion.generate_dataset_json import generate_dataset_json
from multitalent.paths import nnUNet_raw, nnUNet_preprocessed
print(nnUNet_raw)



def convert_epvs(taskname: str, basedir: str, nnunet_dataset_id: int,):
    task_name = taskname

    foldername = "Dataset%03.0d_%s" % (nnunet_dataset_id, task_name)
    val_dir = join(basedir, 'epvs_val/Val/Val')
    art_dir = join(basedir, 'artificial_data')
    basedir = join(basedir, 'epvs_datav3')

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
        if patient.endswith('PVS.nii.gz'):
            case = patient[:-11]
            shutil.copy(join(basedir, case + '_PVS.nii.gz'), join(labelstr, case + '.nii.gz'))
            shutil.copy(join(basedir, case + '_T1w.nii.gz'), join(imagestr, case + '_0000.nii.gz'))
            shutil.copy(join(basedir, case + '_T2w.nii.gz'), join(imagestr, case + '_0001.nii.gz'))
            shutil.copy(join(basedir, case + '_FLAIR.nii.gz'), join(imagestr, case + '_0002.nii.gz'))
            count+=1
    for folder in os.listdir(art_dir):
        if not folder.endswith('.zip'):
            for sub_folder in os.listdir(join(art_dir, folder)):
                for file in os.listdir(join(art_dir, folder, sub_folder, 'SI')):
                    name_parts = file.split('_')
                    case = "_".join(name_parts[:3]) + "_" + "_".join(name_parts[4:6])
                    if name_parts[3] == 'T1w':
                        shutil.copy(join(art_dir, folder, sub_folder, 'SI', file), join(imagestr, case + '_0000.nii.gz'))
                    if name_parts[3] == 'T2w':
                        shutil.copy(join(art_dir, folder, sub_folder, 'SI', file), join(imagestr, case + '_0001.nii.gz'))
                    if name_parts[3] == 'FLAIR':
                        shutil.copy(join(art_dir, folder, sub_folder, 'SI', file), join(imagestr, case + '_0002.nii.gz'))

                        #label
                        label_name1 = join(art_dir, folder, sub_folder, 'PVS_mask_per_case', "_".join(name_parts[:3]) + "_BGPVS_mask_" + "_".join(name_parts[4:]))
                        label_name2 = join(art_dir, folder, sub_folder, 'PVS_mask_per_case', "_".join(name_parts[:3]) + "_BGPVS_mask_" + "_".join(name_parts[4:]))
                        label1_img = sitk.ReadImage(label_name1)
                        label2_img = sitk.ReadImage(label_name2)
                        label_1 = sitk.GetArrayFromImage(label1_img)
                        label_2 = sitk.GetArrayFromImage(label2_img)
                        new_label = np.zeros(np.shape(label_1))
                        new_label[label_1>0.5] = 1
                        new_label[label_2>0.5] = 1

                        new_label_img = sitk.GetImageFromArray(new_label.astype(np.uint8))
                        new_label_img.CopyInformation(label1_img)
                        sitk.WriteImage(new_label_img, join(labelstr, case + '.nii.gz'))
                        count += 1








    for patient in os.listdir(val_dir):
        if not patient.startswith('.DS'):
            in_dir = join(val_dir, patient, patient)
            shutil.copy(join(in_dir + '_T1w.nii.gz'), join(imagests, patient + '_0000.nii.gz'))
            shutil.copy(join(in_dir + '_T2w.nii.gz'), join(imagests, patient + '_0001.nii.gz'))
            shutil.copy(join(in_dir + '_FLAIR.nii.gz'), join(imagests, patient + '_0002.nii.gz'))



    generate_dataset_json(out_base, {"0": "T1", "1": "T2", "2": "Flair"},
                          labels={
                              "background": 0,
                              "EPVS": 1},
                          num_training_cases=count, file_ending='.nii.gz',
                          dataset_name=task_name, reference='epvs_artificial',
                          release='release',
                          overwrite_image_reader_writer='SimpleITKIO',
                          description="brats",
                          license='')


if __name__ == '__main__':
    inpath = '/dkfz/E132-Projekte/Projects/2024_Ulrich_epvs/'
    taskname = 'epvs_artificial'
    #convert_epvs(taskname, inpath, 420)

    split = load_json('/home/c306h/cluster-data/multitalent/nnUNet_preprocessed/Dataset418_epvs_v3/splits_final.json')
    for fold in split:
        for file in os.listdir('/dkfz/E132-Projekte/Projects/2024_Ulrich_collection/challenges/Dataset420_epvs_artificial/labelsTr'):
            if file.startswith('template'):
                fold['train'].append(file[:-7])
    save_json(split,
              '/home/c306h/cluster-data/multitalent/nnUNet_preprocessed/Dataset420_epvs_artificial/splits_final.json')