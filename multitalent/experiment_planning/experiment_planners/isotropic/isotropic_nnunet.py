from typing import Union, List, Tuple

import numpy as np
from multitalent.experiment_planning.experiment_planners.residual_unets.residual_encoder_unet_planners import \
    nnUNetPlannerResEncL, nnUNetPlannerResEncM
from multitalent.preprocessing.normalization.map_channel_name_to_normalization import get_normalization_scheme

class nnUNetPlannerResEncLIso1x1x1(nnUNetPlannerResEncL):
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 24,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'nnUNetResEncUNetLPlansIso1x1x1',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)

    def determine_fullres_target_spacing(self) -> np.ndarray:
        # detect 2D as having shape 1 in the first dimension for all cases
        shapes_after_crop = self.dataset_fingerprint['shapes_after_crop']
        if all([i[0] == 1 for i in shapes_after_crop]):
            return np.array([self.dataset_fingerprint['spacings'][0][0], 1., 1.])
        else:
            return np.array([1., 1., 1.])

    def generate_data_identifier(self, configuration_name: str) -> str:
        return self.plans_identifier + '_' + configuration_name


class nnUNetPlannerResEncLIso(nnUNetPlannerResEncL):
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 24,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'nnUNetResEncUNetLPlansIso',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)

    def determine_fullres_target_spacing(self) -> np.ndarray:
        # compute the spacing according to the old nnUNet rules
        target_spacing = super().determine_fullres_target_spacing()

        # now transfer the lowest spacing to all axes. Do not go lower than the min spacing of each axis though
        target_spacing = np.min(target_spacing)
        spacings = np.vstack(self.dataset_fingerprint['spacings'])
        mins = np.min(spacings, 0)
        target_spacing_here = np.array([max(target_spacing, i) for i in mins])
        return target_spacing_here

    def generate_data_identifier(self, configuration_name: str) -> str:
        return self.plans_identifier + '_' + configuration_name


class nnUNetPlannerResEncLIso2(nnUNetPlannerResEncL):
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 24,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'nnUNetResEncUNetLPlansIso2',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)

    def determine_fullres_target_spacing(self) -> np.ndarray:
        # compute the spacing according to the old nnUNet rules
        target_spacing = super().determine_fullres_target_spacing()

        # now transfer the lowest spacing to all axes. GO as low as you want
        target_spacing = np.min(target_spacing)
        target_spacing_here = np.array([target_spacing] * 3)

        # correct first taxis in case we have a 2D dataset
        if all([i[0] == 1 for i in self.dataset_fingerprint['shapes_after_crop']]):
            target_spacing[0] = self.dataset_fingerprint['spacings'][0][0]
        return target_spacing_here

    def generate_data_identifier(self, configuration_name: str) -> str:
        return self.plans_identifier + '_' + configuration_name

class nnUNetPlannerResEncMIso1x1x1(nnUNetPlannerResEncM):
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 8,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'nnUNetResEncUNetMPlansIso1x1x1',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)

    def determine_fullres_target_spacing(self) -> np.ndarray:
        # detect 2D as having shape 1 in the first dimension for all cases
        shapes_after_crop = self.dataset_fingerprint['shapes_after_crop']
        if all([i[0] == 1 for i in shapes_after_crop]):
            return np.array([self.dataset_fingerprint['spacings'][0][0], 1., 1.])
        else:
            return np.array([1., 1., 1.])

    def generate_data_identifier(self, configuration_name: str) -> str:
        return self.plans_identifier + '_' + configuration_name

class nnUNetPlannerResEncLIso1x1x1_znorm(nnUNetPlannerResEncLIso1x1x1):

    def determine_normalization_scheme_and_whether_mask_is_used_for_norm(self) -> Tuple[List[str], List[bool]]:
        if 'channel_names' not in self.dataset_json.keys():
            print('WARNING: "modalities" should be renamed to "channel_names" in dataset.json. This will be '
                  'enforced soon!')
        modalities = self.dataset_json['channel_names'] if 'channel_names' in self.dataset_json.keys() else \
            self.dataset_json['modality']
        normalization_schemes = [get_normalization_scheme('zscore') for m in modalities.values()]

        if self.dataset_fingerprint['median_relative_size_after_cropping'] < (3 / 4.):
            use_nonzero_mask_for_norm = [i.leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true for i in
                                         normalization_schemes]
        else:
            use_nonzero_mask_for_norm = [False] * len(normalization_schemes)
            assert all([i in (True, False) for i in use_nonzero_mask_for_norm]), 'use_nonzero_mask_for_norm must be ' \
                                                                                 'True or False and cannot be None'
        normalization_schemes = [i.__name__ for i in normalization_schemes]
        return normalization_schemes, use_nonzero_mask_for_norm