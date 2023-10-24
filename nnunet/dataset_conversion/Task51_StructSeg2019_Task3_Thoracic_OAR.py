from batchgenerators.utilities.file_and_folder_operations import *
import shutil
from nnunet.paths import nnUNet_raw_data

if __name__ == "__main__":


    task_id = 51
    task_name = "Task51_StructSeg2019_Task3_Thoracic_OAR"
    base = "/raw_base_dir/"
    out = join(nnUNet_raw_data, task_name)

    data_out = join(out, "imagesTr")
    seg_out = join(out, "labelsTr")
    maybe_mkdir_p(data_out)
    maybe_mkdir_p(seg_out)

    cases = subdirs(base, join=False)

    for c in cases:
        shutil.copy(join(base, c, "data.nii.gz"), join(data_out, c + "_0000.nii.gz"))
        shutil.copy(join(base, c, "label.nii.gz"), join(seg_out, c + ".nii.gz"))

    json_dict = {}
    json_dict['name'] = "StructSeg2019_Task3"
    json_dict['description'] = "https://structseg2019.grand-challenge.org/Dataset/"
    json_dict['tensorImageSize'] = "3D"
    json_dict['reference'] = "https://structseg2019.grand-challenge.org/"
    json_dict['licence'] = "do not touch! Get access from the challenge organizers (https://structseg2019.grand-challenge.org/)"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "CT",
    }
    json_dict['labels'] = {i: str(i) for i in range(1, 7)} # too lazy to type that out
    json_dict['numTraining'] = len(cases)
    json_dict['numTest'] = 0
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in cases]
    json_dict['test'] = []

    with open(os.path.join(out, "dataset.json"), 'w') as f:
        json.dump(json_dict, f, indent=4, sort_keys=True)
