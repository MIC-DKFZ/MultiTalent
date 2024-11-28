from typing import Union, List, Tuple
from torch import nn
from multitalent.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner
from multitalent.preprocessing.normalization.map_channel_name_to_normalization import get_normalization_scheme
from multitalent.preprocessing.normalization.default_normalization_schemes import CTNormalization, NoNormalization, \
    ZScoreNormalization, RescaleTo01Normalization, RGBTo01Normalization, ImageNormalization
class force_zscore_planner(ExperimentPlanner):
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 8,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'force_zscore_defaultplan',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)

    def determine_normalization_scheme_and_whether_mask_is_used_for_norm(self) -> Tuple[List[str], List[bool]]:
        if 'channel_names' not in self.dataset_json.keys():
            print('WARNING: "modalities" should be renamed to "channel_names" in dataset.json. This will be '
                  'enforced soon!')
        modalities = self.dataset_json['channel_names'] if 'channel_names' in self.dataset_json.keys() else \
            self.dataset_json['modality']
        normalization_schemes = len(modalities.values())*[ZScoreNormalization]
        if self.dataset_fingerprint['median_relative_size_after_cropping'] < (3 / 4.):
            use_nonzero_mask_for_norm = [i.leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true for i in
                                         normalization_schemes]
        else:
            use_nonzero_mask_for_norm = [False] * len(normalization_schemes)
            assert all([i in (True, False) for i in use_nonzero_mask_for_norm]), 'use_nonzero_mask_for_norm must be ' \
                                                                                 'True or False and cannot be None'
        normalization_schemes = [i.__name__ for i in normalization_schemes]
        return normalization_schemes, use_nonzero_mask_for_norm