import torch
from multitalent.training.lr_scheduler.polylr import PolyLRScheduler
from multitalent.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from multitalent.architectures.MedNeXt.MedNeXt import MedNeXt

from typing import List, Tuple, Union


class nnUNetTrainer_MedNeXt(nnUNetTrainer):

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-3
        self.num_iterations_per_epoch = 250

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            self.initial_lr,
            weight_decay=self.weight_decay,
            eps=1e-4  # default value 1e-8 might cause nans in fp16
        )

        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler


class nnUNetTrainer_MedNeXt_500_iters(nnUNetTrainer_MedNeXt):

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_iterations_per_epoch = 500


class nnUNetTrainer_MedNeXt_L(nnUNetTrainer_MedNeXt):

    def build_network_architecture(
            architecture_class_name: str,
            arch_init_kwargs: dict,
            arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
            num_input_channels: int,
            num_output_channels: int,
            enable_deep_supervision: bool = True
    ):
        network = MedNeXt(
            in_channels=num_input_channels,
            n_channels=32,
            n_classes=num_output_channels,
            exp_r=[3, 4, 8, 8, 8, 8, 8, 4, 3],
            kernel_size=3,
            deep_supervision=enable_deep_supervision,
            do_res=True,
            do_res_up_down=True,
            block_counts=[3, 4, 8, 8, 8, 8, 8, 4, 3],
            checkpoint_style='outside_block',
            grn=False
        )
        return network


class nnUNetTrainer_MedNeXt_L_GRN(nnUNetTrainer_MedNeXt):

    def build_network_architecture(
            architecture_class_name: str,
            arch_init_kwargs: dict,
            arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
            num_input_channels: int,
            num_output_channels: int,
            enable_deep_supervision: bool = True
    ):
        network = MedNeXt(
            in_channels=num_input_channels,
            n_channels=32,
            n_classes=num_output_channels,
            exp_r=[3, 4, 8, 8, 8, 8, 8, 4, 3],
            kernel_size=3,
            deep_supervision=enable_deep_supervision,
            do_res=True,
            do_res_up_down=True,
            block_counts=[3, 4, 8, 8, 8, 8, 8, 4, 3],
            checkpoint_style='outside_block',
            grn=True
        )
        return network


class nnUNetTrainer_MedNeXt_L_500_iters(nnUNetTrainer_MedNeXt_500_iters):

    def build_network_architecture(
            architecture_class_name: str,
            arch_init_kwargs: dict,
            arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
            num_input_channels: int,
            num_output_channels: int,
            enable_deep_supervision: bool = True
    ):
        network = MedNeXt(
            in_channels=num_input_channels,
            n_channels=32,
            n_classes=num_output_channels,
            exp_r=[3, 4, 8, 8, 8, 8, 8, 4, 3],
            kernel_size=3,
            deep_supervision=enable_deep_supervision,
            do_res=True,
            do_res_up_down=True,
            block_counts=[3, 4, 8, 8, 8, 8, 8, 4, 3],
            checkpoint_style='outside_block',
            grn=False
        )
        return network


class nnUNetTrainer_MedNeXt_L_GRN_500_iters(nnUNetTrainer_MedNeXt_500_iters):

    def build_network_architecture(
            architecture_class_name: str,
            arch_init_kwargs: dict,
            arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
            num_input_channels: int,
            num_output_channels: int,
            enable_deep_supervision: bool = True
    ):
        network = MedNeXt(
            in_channels=num_input_channels,
            n_channels=32,
            n_classes=num_output_channels,
            exp_r=[3, 4, 8, 8, 8, 8, 8, 4, 3],
            kernel_size=3,
            deep_supervision=enable_deep_supervision,
            do_res=True,
            do_res_up_down=True,
            block_counts=[3, 4, 8, 8, 8, 8, 8, 4, 3],
            checkpoint_style='outside_block',
            grn=True
        )
        return network
