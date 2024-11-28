from batchgenerators.utilities.file_and_folder_operations import *
import shutil
import numpy as np
from multitalent.dataset_conversion.generate_dataset_json import generate_dataset_json
from multitalent.paths import nnUNet_raw
print(nnUNet_raw)



def convert_crossmoda(taskname: str, basedir: str, nnunet_dataset_id: int,):
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

    cases = subdirs(basedir, suffix='ceT1.nii.gz', join=False)
    rnd = np.random.RandomState(seed=12345)
    cases = np.sort(cases)
    idx_tr = rnd.choice(len(cases), int(len(cases) * 0.8), replace=False)
    casesTr = [cases[i] for i in idx_tr]
    casesTs = list(set(cases) - set(casesTr))

    for tr in casesTr:
        shutil.copy(join(basedir, tr), join(imagestr, f'{tr[:12]}_0000.nii'))
        shutil.copy(join(basedir, tr[:12] + 'nii.gz'), join(labelstr, f'{tr}.nii'))


    for tr in casesTs:
        shutil.copy(join(basedir, tr), join(imagestr, f'{tr[:12]}_0000.nii'))
        shutil.copy(join(basedir, tr[:12] + 'nii.gz'), join(labelstr, f'{tr}.nii'))

    generate_dataset_json(out_base, {0: "ceT1"},
                          labels={
                              "background": 0,
                              "vestibular schwannoma": 1,
                              "cochlea": 2

                          },
                          num_training_cases=len(casesTr), file_ending='.nii',
                          dataset_name=task_name, reference='',
                          release='',
                          description="Crossmoda22")



inpath = '/home/constantin/Downloads/crossmoda2022_training/training_source/'
taskname = 'crossmoda22_sourceseq'
convert_crossmoda(taskname, inpath, 206)