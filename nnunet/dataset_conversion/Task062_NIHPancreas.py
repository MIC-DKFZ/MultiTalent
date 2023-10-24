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
import os
from collections import OrderedDict
from nnunet.paths import nnUNet_raw_data
from batchgenerators.utilities.file_and_folder_operations import *
import shutil
from multiprocessing import Pool
import nibabel
import dicom2nifti
from nnunet.paths import nnUNet_raw_data


def reorient(filename):
    # print(filename)
    img = nibabel.load(filename)
    img = nibabel.as_closest_canonical(img)
    nibabel.save(img, filename)


if __name__ == "__main__":

    #raw pan ct data
    base = "/home/constantin/Desktop/manifest-1599750808610/Pancreas-CT"

    base_out = "/home/constantin/Desktop/manifest-1599750808610/nifti"
    maybe_mkdir_p(base_out)

    #downloaded label raw data
    labels_base = "/home/constantin/Downloads/TCIA_pancreas_labels-02-05-2017"


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

    # reorient
    p = Pool(4)
    results = []

    for f in subfiles(base_out, suffix=".nii.gz"):
        results.append(p.map_async(reorient, (f, )))
    _ = [i.get() for i in results]

    for f in subfiles(labels_base, suffix=".nii.gz"):
        results.append(p.map_async(reorient, (f, )))
    _ = [i.get() for i in results]

    task_id = 62
    task_name = "NIHPancreas"

    foldername = "Task%03.0d_%s" % (task_id, task_name)

    out_base = join(nnUNet_raw_data, foldername)
    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)

    train_patient_names = []
    test_patient_names = []
    cases = os.listdir(base_out)
    folder_data = base_out
    folder_labels = labels_base
    for c in cases:
        casename =  c[:-7]
        if c not in ['PANCREAS_0045.nii.gz', 'PANCREAS_0007.nii.gz', 'PANCREAS_0032.nii.gz', 'PANCREAS_0027.nii.gz']:
            shutil.copy(join(folder_data, c), join(imagestr, c[:-7] + "_0000.nii.gz"))
            shutil.copy(join(folder_labels, 'label' +c[9:]), join(labelstr, c))
            train_patient_names.append(casename)

    json_dict = OrderedDict()
    json_dict['name'] = task_name
    json_dict['description'] = task_name
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "see website"
    json_dict['licence'] = "see website"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "CT",
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "Pancreas",
    }
    json_dict['numTraining'] = len(train_patient_names)
    json_dict['numTest'] = len(test_patient_names)
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i.split("/")[-1], "label": "./labelsTr/%s.nii.gz" % i.split("/")[-1]} for i in
                             train_patient_names]
    json_dict['test'] = ["./imagesTs/%s.nii.gz" % i.split("/")[-1] for i in test_patient_names]

    save_json(json_dict, os.path.join(out_base, "dataset.json"))
