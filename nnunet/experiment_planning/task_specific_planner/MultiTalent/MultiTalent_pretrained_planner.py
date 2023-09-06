from batchgenerators.utilities.file_and_folder_operations import load_pickle
from nnunet.experiment_planning.experiment_planner_baseline_3DUNet_v21 import ExperimentPlanner3D_v21
from nnunet.paths import *


class ExperimentPlanner3D_v21_Pretrained(ExperimentPlanner3D_v21):
    def __init__(self, folder_with_cropped_data, preprocessed_output_folder, pretrained_model_plans_file: str,
                 pretrained_name: str):
        super().__init__(folder_with_cropped_data, preprocessed_output_folder)
        self.pretrained_model_plans_file = pretrained_model_plans_file
        self.pretrained_name = pretrained_name
        self.data_identifier = "nnUNetData_pretrained_" + pretrained_name
        self.plans_fname = join(self.preprocessed_output_folder, "nnUNetPlans_pretrained_%s_plans_3D.pkl" % pretrained_name)

    def load_pretrained_plans(self):
        classes = self.plans['num_classes']
        self.plans = load_pickle(self.pretrained_model_plans_file)
        self.plans['num_classes'] = classes
        self.transpose_forward = self.plans['transpose_forward']
        self.preprocessor_name = self.plans['preprocessor_name']
        self.plans_per_stage = self.plans['plans_per_stage']
        self.plans['data_identifier'] = self.data_identifier
        self.save_my_plans()
        print(self.plans['plans_per_stage'])

    def run_preprocessing(self, num_threads):
        self.load_pretrained_plans()
        super().run_preprocessing(num_threads)
