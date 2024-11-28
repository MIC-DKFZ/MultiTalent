import argparse
from typing import Union

from batchgenerators.utilities.file_and_folder_operations import join, isdir, isfile, load_json, subfiles, save_json

from multitalent.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json
from multitalent.paths import nnUNet_preprocessed, nnUNet_raw
from multitalent.utilities.file_path_utilities import maybe_convert_to_dataset_name
from multitalent.utilities.plans_handling.plans_handler import PlansManager
from multitalent.utilities.utils import get_filenames_of_train_images_and_targets


def move_plans_between_datasets_onlyconfig(
        source_dataset_name_or_id: Union[int, str],
        target_dataset_name_or_id: Union[int, str],
        source_plans_identifier: str, #old
        target_plans_identifier: str = None,
        normalization: str = None): #new

    target_dataset_name = maybe_convert_to_dataset_name(target_dataset_name_or_id)
    if isfile(source_dataset_name_or_id):
        source_plans = load_json(source_dataset_name_or_id)

    else:
        assert source_plans_identifier!=None, f"Cannot move plans because plans name source dataset is missing. "
        source_dataset_name = maybe_convert_to_dataset_name(source_dataset_name_or_id)

        if target_plans_identifier is None:
            target_plans_identifier = source_plans_identifier

        source_folder = join(nnUNet_preprocessed, source_dataset_name)
        assert isdir(source_folder), f"Cannot move plans because preprocessed directory of source dataset is missing. " \
                                     f"Run multitalent_plan_and_preprocess for source dataset first!"

        source_plans_file = join(source_folder, source_plans_identifier + '.json')
        assert isfile(source_plans_file), f"Source plans are missing. Run the corresponding experiment planning first! " \
                                          f"Expected file: {source_plans_file}"

        source_plans = load_json(source_plans_file)
        assert isfile(join(nnUNet_preprocessed, target_dataset_name, 'nnUNetPlans.json')),  f"Cannot move plans because plan of target dataset is missing. " \
                                     f"Run multitalent_plan_experiment for target dataset first!"
    base_plan = load_json(join(nnUNet_preprocessed, target_dataset_name, 'nnUNetPlans.json'))
    new_plan = load_json(join(nnUNet_preprocessed, target_dataset_name, 'nnUNetPlans.json'))
    new_plan['configurations'] = source_plans['configurations']

    if normalization is None:
        print('Warning: You did not specify the nomalization scheme, each default is used for each datzaset. We recommend to use ZScoreNormalization')
        for config in new_plan['configurations'].keys():
            if config != '3d_cascade_fullres':
                new_plan['configurations'][config]["normalization_schemes"] = base_plan['configurations'][config]["normalization_schemes"]
                new_plan['configurations'][config]["use_mask_for_norm"] = base_plan['configurations'][config]["use_mask_for_norm"]
    else:
        for config in new_plan['configurations'].keys():
            if config != '3d_cascade_fullres':
                new_plan['configurations'][config]["normalization_schemes"] = [normalization]*len(base_plan['configurations'][config]["normalization_schemes"])
                new_plan['configurations'][config]["use_mask_for_norm"] = base_plan['configurations'][config]["use_mask_for_norm"]

    if target_plans_identifier is not None:
        new_plan["plans_name"] = target_plans_identifier
    else:
        new_plan["plans_name"] = source_plans_identifier
        target_plans_identifier = source_plans_identifier

    save_json(new_plan, join(nnUNet_preprocessed, target_dataset_name, target_plans_identifier + '.json'),
              sort_keys=False)


def entry_point_move_plans_between_datasets_onlyconfig():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', type=str, required=True,
                        help='Source dataset name or id or path to a plan file')
    parser.add_argument('-t',nargs='+', type=int,
                        help='Target dataset name or id, can be list')
    parser.add_argument('-sp', type=str, required=False,
                        help='Source plans identifier. If your plans are named "nnUNetPlans.json" then the '
                             'identifier would be nnUNetPlans, not used -s is already filepath')
    parser.add_argument('-tp', type=str, required=False, default=None,
                        help='Target plans identifier. Default is None meaning the source plans identifier will '
                             'be kept. Not recommended if the source plans identifier is a default nnU-Net identifier '
                             'such as nnUNetPlans!!!')
    parser.add_argument('-norm', type=str, required=False, default=None,
                        help='Normalization scheme, can be None if default of each dataset should not be replaced')

    args = parser.parse_args()
    target_d = args.t
    for d in target_d:
        move_plans_between_datasets_onlyconfig(args.s, d, args.sp, args.tp, args.norm)

if __name__ == '__main__':
    # dataset_list = [2,5, 11,24, 27, 35, 38, 101, 127, 201, 204, 205, 206, 207, 221, 223, 701, 233,353, 851]
    # # dataset_list = [203,]
    # for id in dataset_list:
    #     print(id)
    #     move_plans_between_datasets_onlyconfig(3, id, 'nnUNetResEncUNetL1x1x1_Plans_znorm_bs2',)
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', type=str, required=True,
                        help='Source dataset name or id or path to a plan file')
    parser.add_argument('-t',nargs='+', type=int,
                        help='Target dataset name or id, can be list')
    parser.add_argument('-sp', type=str, required=False,
                        help='Source plans identifier. If your plans are named "nnUNetPlans.json" then the '
                             'identifier would be nnUNetPlans, not used -s is already filepath')
    parser.add_argument('-tp', type=str, required=False, default=None,
                        help='Target plans identifier. Default is None meaning the source plans identifier will '
                             'be kept. Not recommended if the source plans identifier is a default nnU-Net identifier '
                             'such as nnUNetPlans!!!')
    parser.add_argument('-norm', type=str, required=False, default=None,
                        help='Normalization scheme, can be None if default of each dataset should not be replaced')

    args = parser.parse_args()
    target_d = args.t
    for d in target_d:
        print(d)
        move_plans_between_datasets_onlyconfig(args.s, d, args.sp, args.tp, args.norm)