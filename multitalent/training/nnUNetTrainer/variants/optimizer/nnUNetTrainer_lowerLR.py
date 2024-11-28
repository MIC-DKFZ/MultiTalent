import torch

from multitalent.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_lr1en4(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 0.0001
