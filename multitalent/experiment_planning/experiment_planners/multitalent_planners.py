from typing import Union, List, Tuple
from multitalent.experiment_planning.experiment_planners.isotropic.isotropic_nnunet import nnUNetPlannerResEncLIso2, nnUNetPlannerResEncLIso1x1x1
from multitalent.preprocessing.resampling.resample_torch import resample_torch_fornnunet


class BIG_Training_nnUNetPlanner_ResEncL_Iso2_torchres(nnUNetPlannerResEncLIso2):
    """
    - iso2
    - torchres
    - resenc L
    """
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 24,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'JHUTraining_nnUNetPlanner_ResEncL_Iso2_torchres',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)

    def generate_data_identifier(self, configuration_name: str) -> str:
        """
        configurations are unique within each plans file but different plans file can have configurations with the
        same name. In order to distinguish the associated data we need a data identifier that reflects not just the
        config but also the plans it originates from
        """
        return self.plans_identifier + '_' + configuration_name

    def determine_resampling(self, *args, **kwargs):
        """
        returns what functions to use for resampling data and seg, respectively. Also returns kwargs
        resampling function must be callable(data, current_spacing, new_spacing, **kwargs)

        determine_resampling is called within get_plans_for_configuration to allow for different functions for each
        configuration
        """
        resampling_data = resample_torch_fornnunet
        resampling_data_kwargs = {
            "is_seg": False,
            'force_separate_z': False,
            'memefficient_seg_resampling': False,
            'num_threads': 8
        }
        resampling_seg = resample_torch_fornnunet
        resampling_seg_kwargs = {
            "is_seg": True,
            'force_separate_z': False,
            'memefficient_seg_resampling': False,
            'num_threads': 8
        }
        return resampling_data, resampling_data_kwargs, resampling_seg, resampling_seg_kwargs
