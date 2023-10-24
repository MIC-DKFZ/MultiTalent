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


from collections import OrderedDict
from nnunet.paths import nnUNet_raw_data
from batchgenerators.utilities.file_and_folder_operations import *
import shutil


if __name__ == "__main__":
    #i used the CervixRawData.zip
    base = "/home/constantin/Downloads/CervixRawData/RawData"

    task_id = 18
    task_name = "PelvicOrganSegmentation"
    prefix = 'ABD'

    foldername = "Task%03.0d_%s" % (task_id, task_name)

    out_base = join(nnUNet_raw_data, foldername)
    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)

    train_folder = join(base, "Training/img")
    label_folder = join(base, "Training/label")
    test_folder = join(base, "Testing/img")
    train_patient_names = []
    test_patient_names = []
    train_patients = subfiles(train_folder, join=False, suffix = 'nii.gz')
    for p in train_patients:
        train_patient_name = p
        label_file = join(label_folder, p[:-13] + '-Mask.nii.gz')
        image_file = join(train_folder, p)
        shutil.copy(image_file, join(imagestr, p[:-7]+'_0000.nii.gz'))
        shutil.copy(label_file, join(labelstr, train_patient_name))
        train_patient_names.append(train_patient_name)

    test_patients = subfiles(test_folder, join=False, suffix=".nii.gz")
    for p in test_patients:
        test_patient_name = p
        image_file = join(test_folder, p)
        shutil.copy(image_file, join(imagests, p[:-7] + '_0000.nii.gz'))
        test_patient_names.append(test_patient_name)

    json_dict = OrderedDict()
    json_dict['name'] = "PelvicOrganSegmentation"
    json_dict['description'] = "Multi-Atlas Labeling Beyond the Cranial Vault Abdominal Organ Segmentation"
    json_dict['tensorImageSize'] = "3D"
    json_dict['reference'] = "https://www.synapse.org/#!Synapse:syn3193805/wiki/217789"
    json_dict['licence'] = "see challenge website"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "CT",
    }
    json_dict['labels'] = OrderedDict({
        "00": "background",
        "01": "bladder",
        "02": "uterus",
        "03": "rectum",
        "04": "small bowel",}
    )
    json_dict['numTraining'] = len(train_patient_names)
    json_dict['numTest'] = len(test_patient_names)
    json_dict['training'] = [{'image': "./imagesTr/%s" % train_patient_name, "label": "./labelsTr/%s" % train_patient_name} for i, train_patient_name in enumerate(train_patient_names)]
    json_dict['test'] = ["./imagesTs/%s" % test_patient_name for test_patient_name in test_patient_names]

    save_json(json_dict, os.path.join(out_base, "dataset.json"))