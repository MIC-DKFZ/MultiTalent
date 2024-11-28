from batchgenerators.utilities.file_and_folder_operations import *
import shutil
import numpy as np
from multitalent.dataset_conversion.generate_dataset_json import generate_dataset_json
from multitalent.paths import nnUNet_raw
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

    cases = subdirs(basedir, prefix='Patient', join=False)
    rnd = np.random.RandomState(seed=12345)
    cases = np.sort(cases)
    idx_tr = rnd.choice(len(cases), int(len(cases) * 0.8), replace=False)
    casesTr = [cases[i] for i in idx_tr]
    casesTs = list(set(cases) - set(casesTr))

    for tr in casesTr:
        for file in os.listdir(join(basedir, tr)):
            if file.endswith(tr[-1]+'-Flair.nii'):
                shutil.copy(join(basedir, tr, file), join(imagestr, f'{tr}_0000.nii'))
            if file.endswith('Seg-Flair.nii'):
                shutil.copy(join(basedir, tr, file), join(labelstr, f'{tr}.nii'))

    for ts in casesTs:
        for file in os.listdir(join(basedir, ts)):
            if file.endswith(ts[-1]+'-Flair.nii'):
                shutil.copy(join(basedir, ts, file), join(imagests, f'{ts}_0000.nii'))
            if file.endswith('Seg-Flair.nii'):
                shutil.copy(join(basedir, ts, file), join(labelsts, f'{ts}.nii'))


    generate_dataset_json(out_base, {0: "Flair"},
                          labels={
                              "background": 0,
                              "MSlesion": 1
                          },
                          num_training_cases=len(casesTr), file_ending='.nii',
                          dataset_name=task_name, reference='',
                          release='release',
                          overwrite_image_reader_writer='NibabelIOWithReorient',
                          description="MS-lesion")



inpath = '/home/constantin/Downloads/8bctsm8jz7-1/Brain MRI Dataset of Multiple Sclerosis with Consensus Manual Lesion Segmentation and Patient Meta Information/'
taskname = 'MS_Flair'
convert_ms(taskname, inpath, 201)