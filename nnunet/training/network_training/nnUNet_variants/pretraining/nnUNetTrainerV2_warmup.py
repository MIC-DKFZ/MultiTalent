#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
from collections import OrderedDict
from batchgenerators.utilities.file_and_folder_operations import *
import torch
from typing import Tuple
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.training.learning_rate.poly_lr import poly_lr
from nnunet.training.dataloading.dataset_loading import unpack_dataset
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from torch.cuda.amp import autocast
from nnunet.training.network_training.nnUNet_variants.transformers.nnUNetTrainerV2_SwinUNETR_ddp import \
    nnUNetTrainerV2_swinunetr_adam_ddp_lr5e4
from nnunet.training.network_training.nnUNet_variants.architectural_variants.nnUNetTrainerV2_ResencUNet_SimonsInit import \
    init_last_bn_before_add_to_0
from nnunet.network_architecture.generic_modular_residual_UNet import FabiansUNet, get_default_network_config
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import distributed


class nnUNetTrainerV2_warmup_increasing_lr(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.warmup_duration = 50
        self.max_num_epochs = 1000 + self.warmup_duration

    def maybe_update_lr(self, epoch=None):
        if self.epoch < self.warmup_duration:
            # epoch 49 is max
            # we increase lr linearly from 0 to initial_lr
            lr = (self.epoch + 1) / self.warmup_duration * self.initial_lr
            self.optimizer.param_groups[0]['lr'] = lr
            self.print_to_log_file("epoch:", self.epoch, "lr:", lr)
        else:
            if epoch is not None:
                ep = epoch - (self.warmup_duration - 1)
            else:
                ep = self.epoch - (self.warmup_duration - 1)
            assert ep > 0, "epoch must be >0"
            return super().maybe_update_lr(ep)

    def on_epoch_end(self) -> bool:
        self.print_to_log_file(self.network.conv_blocks_context[0].blocks[0].conv.weight[0, 0, 0])
        ret = super().on_epoch_end()
        return ret


class nnUNetTrainerV2_warmupsegheads(nnUNetTrainerV2_warmup_increasing_lr):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.num_epochs_sgd_warmup = 50  # lin increase warmup
        self.warmup_max_lr = 5e-4  # for heads
        self.warmup_duration = 10  # this is for the seg heads
        self.max_num_epochs = 1000 + self.num_epochs_sgd_warmup + self.warmup_duration

    # def process_plans(self, plans):
    #     super().process_plans(plans)
    #     self.patch_size = [64,96,96]
    def initialize(self, training=True, force_load_plans=False):
        # here we call initialize_optimizer_and_scheduler with seg heads only
        super().initialize(training, force_load_plans)
        if training:
            self.initialize_optimizer_and_scheduler(True)

    def maybe_update_lr(self, epoch=None):
        print(self.warmup_duration)
        if self.epoch < self.warmup_duration:
            # we increase lr linearly from 0 to self.warmup_max_lr
            lr = (self.epoch + 1) / self.warmup_duration * self.warmup_max_lr
            self.optimizer.param_groups[0]['lr'] = lr
            self.lr = lr
            self.print_to_log_file("epoch:", self.epoch, "lr for heads:", lr)
        elif self.warmup_duration <= self.epoch < self.warmup_duration + self.num_epochs_sgd_warmup:
            # we increase lr linearly from 0 to self.initial_lr
            lr = (self.epoch - self.warmup_duration + 1) / self.num_epochs_sgd_warmup * self.initial_lr
            self.optimizer.param_groups[0]['lr'] = lr
            self.print_to_log_file("epoch:", self.epoch, "lr now lin increasing whole network:", lr)
        else:
            if epoch is not None:
                ep = epoch - (self.warmup_duration + self.num_epochs_sgd_warmup - 1)
            else:
                ep = self.epoch - (self.warmup_duration + self.num_epochs_sgd_warmup - 1)
            assert ep > 0, "epoch must be >0"

            self.optimizer.param_groups[0]['lr'] = poly_lr(ep,
                                                           self.max_num_epochs - self.num_epochs_sgd_warmup - self.warmup_duration,
                                                           self.initial_lr, 0.9)
            self.print_to_log_file("lr was set to:", np.round(self.optimizer.param_groups[0]['lr'], decimals=6))

    def on_epoch_end(self) -> bool:
        self.print_to_log_file(self.network.conv_blocks_context[0].blocks[0].conv.weight[0, 0, 0])
        if self.epoch == self.warmup_duration:
            self.print_to_log_file("now train whole network")
            self.initialize_optimizer_and_scheduler(seg_heads_only=False)
        ret = super().on_epoch_end()
        return ret

    def initialize_optimizer_and_scheduler(self, seg_heads_only=False):
        assert self.network is not None, "self.initialize_network must be called first"

        if seg_heads_only:
            parameters = self.network.seg_outputs.parameters()
            self.optimizer = torch.optim.AdamW(parameters, 3e-3, weight_decay=self.weight_decay, amsgrad=True)
        else:
            parameters = self.network.parameters()
            self.optimizer = torch.optim.SGD(parameters, self.initial_lr, weight_decay=self.weight_decay,
                                             momentum=0.99, nesterov=True)

        self.lr_scheduler = None

    def load_checkpoint_ram(self, checkpoint, train=True):
        """
        we need to have the correct parameters in the optimizer  (warmup etc)

        :param checkpoint:
        :param train:
        :return:
        """
        if not self.was_initialized:
            self.initialize(train)

        new_state_dict = OrderedDict()
        curr_state_dict_keys = list(self.network.state_dict().keys())
        # if state dict comes form nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        for k, value in checkpoint['state_dict'].items():
            key = k

            if key not in curr_state_dict_keys and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value

        if self.fp16:
            self._maybe_init_amp()
            if 'amp_grad_scaler' in checkpoint.keys():
                self.amp_grad_scaler.load_state_dict(checkpoint['amp_grad_scaler'])

        self.network.load_state_dict(new_state_dict)
        self.epoch = checkpoint['epoch']
        if train:
            # we need to have the correct parameters in the optimizer
            if self.epoch > self.warmup_duration: self.initialize_optimizer_and_scheduler(seg_heads_only=False)

            optimizer_state_dict = checkpoint['optimizer_state_dict']
            if optimizer_state_dict is not None:
                self.optimizer.load_state_dict(optimizer_state_dict)

            if self.lr_scheduler is not None and hasattr(self.lr_scheduler, 'load_state_dict') and checkpoint[
                'lr_scheduler_state_dict'] is not None:
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

            if issubclass(self.lr_scheduler.__class__, _LRScheduler):
                self.lr_scheduler.step(self.epoch)

        self.all_tr_losses, self.all_val_losses, self.all_val_losses_tr_mode, self.all_val_eval_metrics = checkpoint[
            'plot_stuff']

        # load best loss (if present)
        if 'best_stuff' in checkpoint.keys():
            self.best_epoch_based_on_MA_tr_loss, self.best_MA_tr_loss_for_patience, self.best_val_eval_criterion_MA = \
                checkpoint[
                    'best_stuff']

        # after the training is done, the epoch is incremented one more time in my old code. This results in
        # self.epoch = 1001 for old trained models when the epoch is actually 1000. This causes issues because
        # len(self.all_tr_losses) = 1000 and the plot function will fail. We can easily detect and correct that here
        if self.epoch != len(self.all_tr_losses):
            self.print_to_log_file("WARNING in loading checkpoint: self.epoch != len(self.all_tr_losses). This is "
                                   "due to an old bug and should only appear when you are loading old models. New "
                                   "models should have this fixed! self.epoch is now set to len(self.all_tr_losses)")
            self.epoch = len(self.all_tr_losses)
            self.all_tr_losses = self.all_tr_losses[:self.epoch]
            self.all_val_losses = self.all_val_losses[:self.epoch]
            self.all_val_losses_tr_mode = self.all_val_losses_tr_mode[:self.epoch]
            self.all_val_eval_metrics = self.all_val_eval_metrics[:self.epoch]

        self._maybe_init_amp()


class nnUNetTrainerV2_warmupsegheads_swinunetr_adam_lr5e4_ddp(nnUNetTrainerV2_swinunetr_adam_ddp_lr5e4):
    def __init__(self, plans_file, fold, local_rank, output_folder=None, dataset_directory=None, batch_dice=True,
                 stage=None, unpack_data=True, deterministic=True, distribute_batch_size=False, fp16=False):
        super().__init__(plans_file, fold, local_rank, output_folder=output_folder, dataset_directory=dataset_directory,
                         batch_dice=batch_dice,
                         stage=stage, unpack_data=unpack_data, deterministic=deterministic,
                         distribute_batch_size=distribute_batch_size, fp16=fp16)
        self.num_epochs_sgd_warmup = 50  # lin increase warmup
        self.warmup_max_lr = 1e-4  # for heads
        self.warmup_duration = 10  # this is for the seg heads
        self.max_num_epochs = 1000 + self.num_epochs_sgd_warmup + self.warmup_duration
        self.initial_lr = 5e-4

    def initialize(self, training=True, force_load_plans=False):

        """
        :param training:
        :return:
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)

            self.setup_DA_params()

            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)
            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    if self.local_rank == 0:
                        print("unpacking dataset")
                        unpack_dataset(self.folder_with_preprocessed_data)
                        print("done")
                    distributed.barrier()
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")
                seeds_train = np.random.random_integers(0, 99999, self.data_aug_params.get('num_threads'))
                seeds_val = np.random.random_integers(0, 99999, max(self.data_aug_params.get('num_threads') // 2, 1))
                print("seeds train", seeds_train)
                print("seeds_val", seeds_val)
                assert self.deep_supervision_scales is None
                self.tr_gen, self.val_gen = get_moreDA_augmentation(self.dl_tr, self.dl_val,
                                                                    self.data_aug_params[
                                                                        'patch_size_for_spatialtransform'],
                                                                    self.data_aug_params,
                                                                    deep_supervision_scales=self.deep_supervision_scales,
                                                                    seeds_train=seeds_train,
                                                                    seeds_val=seeds_val,
                                                                    pin_memory=self.pin_memory)
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler(True)
            self.network = DDP(self.network, device_ids=[self.local_rank])

            assert isinstance(self.network, DDP)
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

        # self.network.module.out.conv.conv.weights.data.uniform_(0.0, 1.0)
        # self.network.module.out.conv.conv.bias.data.fill_(0)

    # def process_plans(self, plans):
    #     super().process_plans(plans)
    #     self.patch_size = [96,96,96]
    #     self.batch_size = 2
    def maybe_update_lr(self, epoch=None):
        if self.epoch < self.warmup_duration:
            # we increase lr linearly from 0 to self.warmup_max_lr
            lr = (self.epoch + 1) / self.warmup_duration * self.warmup_max_lr
            self.optimizer.param_groups[0]['lr'] = lr
            self.print_to_log_file("epoch:", self.epoch, "lr for heads:", lr)

        elif self.warmup_duration <= self.epoch < self.warmup_duration + self.num_epochs_sgd_warmup:
            # we increase lr linearly from 0 to self.initial_lr
            lr = (self.epoch - self.warmup_duration + 1) / self.num_epochs_sgd_warmup * self.initial_lr
            self.optimizer.param_groups[0]['lr'] = lr
            self.print_to_log_file("epoch:", self.epoch, "lr now lin increasing whole network:", lr)
        else:
            if epoch is not None:
                ep = epoch - (self.warmup_duration + self.num_epochs_sgd_warmup - 1)
            else:
                ep = self.epoch - (self.warmup_duration + self.num_epochs_sgd_warmup - 1)
            assert ep > 0, "epoch must be >0"

            self.optimizer.param_groups[0]['lr'] = poly_lr(ep,
                                                           self.max_num_epochs - self.num_epochs_sgd_warmup - self.warmup_duration,
                                                           self.initial_lr, 0.9)
            self.print_to_log_file("lr was set to:", np.round(self.optimizer.param_groups[0]['lr'], decimals=6))

    def on_epoch_end(self) -> bool:
        self.print_to_log_file(self.network.module.encoder1.layer.conv1.conv.weight[0, 0, 0])
        self.print_to_log_file(self.network.module.out.conv[0].weight[0, 0, 0])
        if self.epoch == self.warmup_duration:
            self.print_to_log_file("now train whole network")
            self.initialize_optimizer_and_scheduler(seg_heads_only=False)
        ret = nnUNetTrainerV2.on_epoch_end(self)
        return ret

    def initialize_optimizer_and_scheduler(self, seg_heads_only=False):
        assert self.network is not None, "self.initialize_network must be called first"

        if seg_heads_only:
            if isinstance(self.network, DDP):
                parameters = self.network.module.out.parameters()
            else:
                parameters = self.network.out.parameters()
            self.optimizer = torch.optim.AdamW(parameters, weight_decay=self.weight_decay, eps=1e-4)
        else:
            parameters = self.network.module.parameters()
            self.optimizer = torch.optim.AdamW(parameters,
                                               self.initial_lr,
                                               weight_decay=self.weight_decay,
                                               eps=1e-4)  # 1e-8 might cause nans in fp16

        self.lr_scheduler = None

    def load_checkpoint_ram(self, checkpoint, train=True):
        """
        we need to have the correct parameters in the optimizer  (warmup etc)

        :param checkpoint:
        :param train:
        :return:
        """
        if not self.was_initialized:
            self.initialize(train)

        new_state_dict = OrderedDict()
        curr_state_dict_keys = list(self.network.state_dict().keys())
        # if state dict comes form nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        for k, value in checkpoint['state_dict'].items():
            key = k

            if key not in curr_state_dict_keys and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value

        if self.fp16:
            self._maybe_init_amp()
            if 'amp_grad_scaler' in checkpoint.keys():
                self.amp_grad_scaler.load_state_dict(checkpoint['amp_grad_scaler'])

        self.network.load_state_dict(new_state_dict)
        self.epoch = checkpoint['epoch']
        if train:
            # we need to have the correct parameters in the optimizer
            if self.epoch > self.warmup_duration: self.initialize_optimizer_and_scheduler(seg_heads_only=False)

            optimizer_state_dict = checkpoint['optimizer_state_dict']
            if optimizer_state_dict is not None:
                self.optimizer.load_state_dict(optimizer_state_dict)

            if self.lr_scheduler is not None and hasattr(self.lr_scheduler, 'load_state_dict') and checkpoint[
                'lr_scheduler_state_dict'] is not None:
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

            if issubclass(self.lr_scheduler.__class__, _LRScheduler):
                self.lr_scheduler.step(self.epoch)

        self.all_tr_losses, self.all_val_losses, self.all_val_losses_tr_mode, self.all_val_eval_metrics = checkpoint[
            'plot_stuff']

        # load best loss (if present)
        if 'best_stuff' in checkpoint.keys():
            self.best_epoch_based_on_MA_tr_loss, self.best_MA_tr_loss_for_patience, self.best_val_eval_criterion_MA = \
                checkpoint[
                    'best_stuff']

        # after the training is done, the epoch is incremented one more time in my old code. This results in
        # self.epoch = 1001 for old trained models when the epoch is actually 1000. This causes issues because
        # len(self.all_tr_losses) = 1000 and the plot function will fail. We can easily detect and correct that here
        if self.epoch != len(self.all_tr_losses):
            self.print_to_log_file("WARNING in loading checkpoint: self.epoch != len(self.all_tr_losses). This is "
                                   "due to an old bug and should only appear when you are loading old models. New "
                                   "models should have this fixed! self.epoch is now set to len(self.all_tr_losses)")
            self.epoch = len(self.all_tr_losses)
            self.all_tr_losses = self.all_tr_losses[:self.epoch]
            self.all_val_losses = self.all_val_losses[:self.epoch]
            self.all_val_losses_tr_mode = self.all_val_losses_tr_mode[:self.epoch]
            self.all_val_eval_metrics = self.all_val_eval_metrics[:self.epoch]

        self._maybe_init_amp()


    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data, gpu_id=None)
            target = to_cuda(target, gpu_id=None)

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                output = self.network(data)
                del data
                l = self.compute_loss(output, target)

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                if self.epoch > self.warmup_duration:
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)

                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            del data
            l = self.compute_loss(output, target)

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target

        return l.detach().cpu().numpy()

class nnUNetTrainerV2_warmupsegheads_resenc(nnUNetTrainerV2_warmupsegheads):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.num_epochs_sgd_warmup = 50  # lin increase warmup
        self.warmup_max_lr = 5e-4  # for heads
        self.warmup_duration = 10  # this is for the seg heads
        self.max_num_epochs = 1000 + self.num_epochs_sgd_warmup + self.warmup_duration

    def initialize_network(self):
        if self.threeD:
            cfg = get_default_network_config(3, None, norm_type="in")

        else:
            cfg = get_default_network_config(1, None, norm_type="in")

        stage_plans = self.plans['plans_per_stage'][self.stage]
        conv_kernel_sizes = stage_plans['conv_kernel_sizes']
        blocks_per_stage_encoder = stage_plans['num_blocks_encoder']
        blocks_per_stage_decoder = stage_plans['num_blocks_decoder']
        pool_op_kernel_sizes = stage_plans['pool_op_kernel_sizes']

        self.network = FabiansUNet(self.num_input_channels, self.base_num_features, blocks_per_stage_encoder, 2,
                                   pool_op_kernel_sizes, conv_kernel_sizes, cfg, self.num_classes,
                                   blocks_per_stage_decoder, True, False, 320, InitWeights_He(1e-2))

        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper
        self.network.apply(init_last_bn_before_add_to_0)

    def initialize_optimizer_and_scheduler(self, seg_heads_only=False):
        assert self.network is not None, "self.initialize_network must be called first"
        self.print_to_log_file(self.network.decoder.deep_supervision_outputs[0].weight[0, 0, 0])
        if seg_heads_only:
            parameters = self.network.decoder.deep_supervision_outputs.parameters()
            # for param in self.network.decoder.deep_supervision_outputs.parameters():
            #     print(type(param), param.size())
            self.optimizer = torch.optim.AdamW(parameters, 3e-3, weight_decay=self.weight_decay, amsgrad=True)
        else:
            parameters = self.network.parameters()
            # for param in self.network.parameters():
            #     print(type(param), param.size())
            self.optimizer = torch.optim.SGD(parameters, self.initial_lr, weight_decay=self.weight_decay,
                                             momentum=0.99, nesterov=True)

        self.lr_scheduler = None

    def setup_DA_params(self):
        """
        net_num_pool_op_kernel_sizes is different in resunet
        """
        super().setup_DA_params()
        self.deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
            np.vstack(self.net_num_pool_op_kernel_sizes[1:]), axis=0))[:-1]

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True, step_size: float = 0.5,
                 save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 force_separate_z: bool = None, interpolation_order: int = 3, interpolation_order_z=0,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):

        ds = self.network.decoder.deep_supervision
        self.network.decoder.deep_supervision = False
        ret = nnUNetTrainer.validate(self, do_mirroring=do_mirroring, use_sliding_window=use_sliding_window,
                                     step_size=step_size, save_softmax=save_softmax, use_gaussian=use_gaussian,
                                     overwrite=overwrite, validation_folder_name=validation_folder_name, debug=debug,
                                     all_in_gpu=all_in_gpu, segmentation_export_kwargs=segmentation_export_kwargs,
                                     run_postprocessing_on_folds=run_postprocessing_on_folds)
        self.network.decoder.deep_supervision = ds

        return ret

    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                         verbose: bool = True, mixed_precision=True) -> Tuple[
        np.ndarray, np.ndarray]:
        ds = self.network.decoder.deep_supervision
        self.network.decoder.deep_supervision = False
        ret = nnUNetTrainer.predict_preprocessed_data_return_seg_and_softmax(self, data, do_mirroring=do_mirroring,
                                                                             mirror_axes=mirror_axes,
                                                                             use_sliding_window=use_sliding_window,
                                                                             step_size=step_size,
                                                                             use_gaussian=use_gaussian,
                                                                             pad_border_mode=pad_border_mode,
                                                                             pad_kwargs=pad_kwargs,
                                                                             all_in_gpu=all_in_gpu,
                                                                             verbose=verbose,
                                                                             mixed_precision=mixed_precision)
        self.network.decoder.deep_supervision = ds
        return ret

    def run_training(self):
        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
        # want at the start of the training
        ds = self.network.decoder.deep_supervision
        self.network.decoder.deep_supervision = True
        ret = nnUNetTrainer.run_training(self)
        self.network.decoder.deep_supervision = ds
        return ret

    def on_epoch_end(self) -> bool:
        self.print_to_log_file(self.network.encoder.stages[0].convs[0].conv1.weight[0, 0, 0])
        self.print_to_log_file(self.network.decoder.deep_supervision_outputs[0].weight[0, 0, 0])
        if self.epoch == self.warmup_duration:
            self.print_to_log_file("now train whole network")
            self.initialize_optimizer_and_scheduler(seg_heads_only=False)
        ret = nnUNetTrainerV2.on_epoch_end(self)
        return ret

    # def process_plans(self, plans):
    # super().process_plans(plans)
    # self.patch_size = [96,96,96]
    # self.batch_size = 2

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """

        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']
        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                output = self.network(data)
                del data
                l = self.loss(output, target)
                # print(l)
                # print(torch.isnan(output[0]).any())
            if do_backprop:
                # print('not skipped')
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                if self.epoch >= 10:
                    clip = 12
                else:
                    clip = 1
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), clip)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            del data
            l = self.loss(output, target)

            if do_backprop:
                l.backward()
                if self.epoch > 10:
                    clip = 12
                else:
                    clip = 4
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), clip)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target

        return l.detach().cpu().numpy()


