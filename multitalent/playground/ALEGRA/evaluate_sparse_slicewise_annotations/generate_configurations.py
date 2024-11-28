from multitalent.playground.ALEGRA.evaluate_sparse_slicewise_annotations.configuration import *
from batchgenerators.utilities.file_and_folder_operations import *
from multitalent.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from multitalent.paths import nnUNet_preprocessed

if __name__ == '__main__':
    for d in datasets:
        dataset_name = maybe_convert_to_dataset_name(d)
        plans_file = join(nnUNet_preprocessed, dataset_name, 'nnUNetPlans.json')
        plans = load_json(plans_file)
        for nc, pp in new_configurations.items():
            plans['configurations'][nc] = {
                'inherits_from': base_configuration,
                "data_identifier": 'nnUNetPlans_' + pp,
                "preprocessor_name": pp
            }
        write_json(plans, plans_file)
