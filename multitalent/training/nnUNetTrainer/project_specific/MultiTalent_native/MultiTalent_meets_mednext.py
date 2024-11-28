from typing import Union, Tuple, List
from multitalent.training.nnUNetTrainer.project_specific.MultiTalent_native.MultiTalent_multistem import MultiTalent_trainer_multistems_4000ep
from multitalent.utilities.MultiTalent.MedNeXt_meets_MultiTalent import MedNeXt_for_MT
import torch
import os
from torch import autocast, nn
from torch._dynamo import OptimizedModule
from multitalent.training.lr_scheduler.polylr import PolyLRScheduler
from multitalent.utilities.helpers import empty_cache, dummy_context

class MultiTalent_meets_mednext_trainer_multistem_4000ep(MultiTalent_trainer_multistems_4000ep):

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
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int|dict,
                                   num_output_channels: dict,
                                   enable_deep_supervision: bool = True) -> nn.Module:


        network = MedNeXt_for_MT(
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

    def set_deep_supervision_enabled(self, enabled: bool):
        """
        This function is specific for the default architecture in nnU-Net. If you change the architecture, there are
        chances you need to change this as well!
        """
        if self.is_ddp:
            mod = self.network.module
        else:
            mod = self.network
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod

        mod.deep_supervision = enabled

    def _do_i_compile(self):
        return False
        # # # # new default: compile is enabled!
        # # # # not for MT
        # if 'nnUNet_compile' not in os.environ.keys():
        #    return True
        # else:
        #    return os.environ['nnUNet_compile'].lower() in ('true', '1', 't')

    def train_step(self, batch: list[dict], ids:list) -> dict:
        data = []
        b_target = []
        for i in range(self.batch_size):
            data.append(batch[i]['data'])

            if isinstance(batch[i]['seg'], list):
                target = [i.to(self.device, non_blocking=True) for i in batch[i]['seg']]
            else:
                target = batch[i]['seg'].to(self.device, non_blocking=True)
            b_target.append(target)


        data = [i.to(self.device, non_blocking=True) for i in data]

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            # print(ids, data.shape, torch.max(data[0,1]))
            for b in range(len(b_target)):
                if torch.isnan(data[b]).any():
                    print(batch[b]['keys'], 'data nan data')
                if torch.isnan(b_target[b][0]).any():
                    print(batch[b]['keys'], 'target nan data')
            output = self.network(data, ids)
            # print(len(output[0]), 'output_len')
            # print(len(b_target[0]), 'target_len')
            # print(output[0][0].shape,  'output_shape')
            # print(b_target[0][0].shape, 'target_shape')
            # del data
            for c,id in enumerate(ids):
                if c == 0:
                    l = self.loss[id](output[c], b_target[c])
                else:
                    l += self.loss[id](output[c], b_target[c])
        if torch.isnan(l):
            l = torch.zeros( 1 , device = l.device)
            print([batch[i]['keys'] for i in range(len(batch))], 'nan loss')
            return {'loss': l.detach().cpu().numpy()}
        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}