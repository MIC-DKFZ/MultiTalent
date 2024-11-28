from __future__ import annotations
from multitalent.training.nnUNetTrainer.project_specific.MultiTalent_native.MultiTalent_trainer import MultiTalent_trainer
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from multitalent.utilities.label_handling.label_handling import determine_num_input_channels




class MultiTalent_trainer_multistems(MultiTalent_trainer):

    def check_MT_plans(self):
        # checks if plan configurations are the same, just some props
        what_to_check = ['patch_size', 'spacing', 'architecture' ]

        configuration_props = vars(self.plans_manager.get_configuration(self.configuration_name))
        for id in self.all_ids:
            cur_props = vars(self.plans_managers[id].get_configuration(self.configuration_name))
            for prop in what_to_check:
                if configuration_props['configuration'][prop] != cur_props['configuration'][prop]:
                    print(f"Plan configurations of main plan and plan {id} are different in:", prop)
                    return False
        return True
    def initialize(self):
        if not self.was_initialized:
            # now we need the sampling prob for each dataset (if more then x classes:
            # dataset sampling prob is devided for these subgroups)
            # certainly smarter ways but its already late.....
            all_img = np.sum([self.num_cases_per_dataset[i] for i in self.num_cases_per_dataset.keys()])
            self.print_to_log_file('number of all images: %d' % all_img)
            prop_per_case_in_d = {}
            props_case_normed = {}
            for i in self.num_cases_per_dataset.keys():
                prop_per_case_in_d[i] = 1 / np.sqrt(self.num_cases_per_dataset[i])
            for j in prop_per_case_in_d.keys():
                props_case_normed[j] = prop_per_case_in_d[j] / np.sum(
                    [prop_per_case_in_d[i] * self.num_cases_per_dataset[i] for i in self.num_cases_per_dataset.keys()])

            self.sample_probs = {}
            for id in self.all_ids:
                raw_id = id.split('_')[0]

                # if datset has to many classes
                datset_parts = 0
                for key in self.all_ids:
                    if key.startswith(raw_id + "_"):
                        datset_parts += 1
                if datset_parts == 0:
                    datset_parts = 1
                self.sample_probs[id] = props_case_normed[raw_id] * self.num_cases_per_dataset[raw_id] / datset_parts

            for id in self.all_ids:
                self.print_to_log_file('Prob for Dataset %s is %f' % (id, self.sample_probs[id]))

            # we want MT to allow to have different channel inputs
            self.print_to_log_file('max_input_channels: ', self.num_input_channels)

            self.all_num_seg_heads = {}
            for id in self.all_ids:
                # custumn change for the datasets that have a high number of classes to not run in OOM GPU
                if '_' in id:
                    self.all_num_seg_heads[id] = len(self.labelmapping[id]) + 1

                else:
                    self.all_num_seg_heads[id] = self.label_managers[id].num_segmentation_heads

            # critical part for MT, but just need to update self.build_network_architecture

            self.network = self.build_network_architecture(
                self.configuration_manager.network_arch_class_name,
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.input_channels,
                self.all_num_seg_heads,
                self.enable_deep_supervision
            ).to(self.device)

            # compile network for free speedup
            if self._do_i_compile():
                self.print_to_log_file('Compiling network...')
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank], find_unused_parameters=True)

            # needs to be adapted for MT
            self.loss = {}
            for id in self.all_ids:
                self.loss[id] = self._build_loss(id)

            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")


class MultiTalent_trainer_multistems_4000ep(MultiTalent_trainer_multistems):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 4000