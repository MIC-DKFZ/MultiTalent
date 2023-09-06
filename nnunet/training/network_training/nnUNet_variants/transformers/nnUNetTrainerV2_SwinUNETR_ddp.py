'''
Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.'''

from __future__ import annotations
import torch
from nnunet.utilities.tensor_utilities import sum_tensor
from nnunet.training.network_training.nnUNetTrainerV2_DDP import nnUNetTrainerV2_DDP
from nnunet.utilities.nd_softmax import softmax_helper
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from collections import OrderedDict
from sklearn.model_selection import KFold
from _warnings import warn
from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from nnunet.training.dataloading.dataset_loading import unpack_dataset
from nnunet.training.loss_functions.dice_loss import get_tp_fp_fn_tn
from nnunet.utilities.distributed import awesome_allgather_function
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn, distributed

from nnunet.training.data_augmentation.default_data_augmentation import default_3D_augmentation_params, \
    default_2D_augmentation_params, get_patch_size
from monai.networks.nets.swin_unetr import SwinUNETR as SwinUNETR_Orig

from nnunet.network_architecture.neural_network import SegmentationNetwork

# Wrapper using SegmentationNetwork object. To pass nnUNet internal check
class SwinUNETR(SwinUNETR_Orig, SegmentationNetwork):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Segmentation Network Params. Needed for the nnUNet evaluation pipeline
        self.conv_op = nn.Conv3d
        # self.inference_apply_nonlin = softmax_helper
        self.input_shape_must_be_divisible_by = 16  # just some random val 2**5
        self.num_classes = kwargs['out_channels']
        self.do_ds = False



class nnUNetTrainerV2_swinunetr_adam_ddp(nnUNetTrainerV2_DDP):
    def __init__(self, plans_file, fold, local_rank, output_folder=None, dataset_directory=None, batch_dice=True,
                 stage=None, unpack_data=True, deterministic=True, distribute_batch_size=False, fp16=False):
        super().__init__(plans_file, fold, local_rank, output_folder=output_folder, dataset_directory=dataset_directory, batch_dice=batch_dice,
                 stage=stage, unpack_data=unpack_data, deterministic=deterministic, distribute_batch_size=distribute_batch_size, fp16=fp16)

        self.initial_lr = 1e-3

    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.pkl file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.pkl file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        splits_file = join(self.dataset_directory, "splits_final.pkl")

        # if the split file does not exist we need to create it
        if not isfile(splits_file):
            self.print_to_log_file("Creating new 5-fold cross-validation split...")
            splits = []
            all_keys_sorted = np.sort(list(self.dataset.keys()))
            kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
            for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                train_keys = np.array(all_keys_sorted)[train_idx]
                test_keys = np.array(all_keys_sorted)[test_idx]
                splits.append(OrderedDict())
                splits[-1]['train'] = train_keys
                splits[-1]['val'] = test_keys
            save_pickle(splits, splits_file)

        else:
            self.print_to_log_file("Using splits from existing split file:", splits_file)
            splits = load_pickle(splits_file)
            self.print_to_log_file("The split file contains %d splits." % len(splits))

        self.print_to_log_file("Desired fold for training: %d" % self.fold)
        if self.fold < len(splits):
            tr_keys = splits[self.fold]['train']
            val_keys = splits[self.fold]['val']
            self.print_to_log_file("This split has %d training and %d validation cases."
                                   % (len(tr_keys), len(val_keys)))
        else:
            self.print_to_log_file("INFO: You requested fold %d for training but splits "
                                   "contain only %d folds. I am now creating a "
                                   "random (but seeded) 80:20 split!" % (self.fold, len(splits)))
            # if we request a fold that is not in the split file, create a random 80:20 split
            rnd = np.random.RandomState(seed=12345 + self.fold)
            keys = np.sort(list(self.dataset.keys()))
            idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
            idx_val = [i for i in range(len(keys)) if i not in idx_tr]
            tr_keys = [keys[i] for i in idx_tr]
            val_keys = [keys[i] for i in idx_val]
            self.print_to_log_file("This random 80:20 split has %d training and %d validation cases."
                                   % (len(tr_keys), len(val_keys)))

        tr_keys.sort()
        val_keys.sort()
        self.dataset_tr = OrderedDict()
        for i in tr_keys:
            self.dataset_tr[i] = self.dataset[i]
        self.dataset_val = OrderedDict()
        for i in val_keys:
            self.dataset_val[i] = self.dataset[i]

    def initialize_network(self):
        """
        changed deep supervision to False
        :return:
        """
        self.network = SwinUNETR(img_size=(96, 192, 192),
                            in_channels=self.num_input_channels,
                            out_channels=self.num_classes,
                            feature_size=48,
                            use_checkpoint=False,)
        # self.network = Generic_UNet(self.num_input_channels, self.base_num_features, self.num_classes,
        #                             len(self.net_num_pool_op_kernel_sizes),
        #                             self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
        #                             net_nonlin, net_nonlin_kwargs, False, False, lambda x: x, InitWeights_He(1e-2),
        #                             self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
        if torch.cuda.is_available():
            self.network.cuda()
        # self.network.inference_apply_nonlin = softmax_helper

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.AdamW(self.network.parameters(),
                                            self.initial_lr,
                                            weight_decay=self.weight_decay,
                                            eps=1e-4        # 1e-8 might cause nans in fp16
                                        )
        self.lr_scheduler = None

    def setup_DA_params(self):
        """
        we leave out the creation of self.deep_supervision_scales, so it remains None
        :return:
        """
        if self.threeD:
            self.data_aug_params = default_3D_augmentation_params
            self.data_aug_params['rotation_x'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_y'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_z'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            if self.do_dummy_2D_aug:
                self.data_aug_params["dummy_2D"] = True
                self.print_to_log_file("Using dummy2d data augmentation")
                self.data_aug_params["elastic_deform_alpha"] = \
                    default_2D_augmentation_params["elastic_deform_alpha"]
                self.data_aug_params["elastic_deform_sigma"] = \
                    default_2D_augmentation_params["elastic_deform_sigma"]
                self.data_aug_params["rotation_x"] = default_2D_augmentation_params["rotation_x"]
        else:
            self.do_dummy_2D_aug = False
            if max(self.patch_size) / min(self.patch_size) > 1.5:
                default_2D_augmentation_params['rotation_x'] = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            self.data_aug_params = default_2D_augmentation_params
        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm

        if self.do_dummy_2D_aug:
            self.basic_generator_patch_size = get_patch_size(self.patch_size[1:],
                                                             self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            self.basic_generator_patch_size = np.array([self.patch_size[0]] + list(self.basic_generator_patch_size))
        else:
            self.basic_generator_patch_size = get_patch_size(self.patch_size, self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])

        self.data_aug_params["scale_range"] = (0.7, 1.4)
        self.data_aug_params["do_elastic"] = False
        self.data_aug_params['selected_seg_channels'] = [0]
        self.data_aug_params['patch_size_for_spatialtransform'] = self.patch_size


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
            self.initialize_optimizer_and_scheduler()
            self.maybe_update_lr()
            self.network = DDP(self.network, device_ids=[self.local_rank])

        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

    def process_plans(self, plans):
        super().process_plans(plans)
        # self.patch_size = [64, 64, 64]


    def run_online_evaluation(self, output, target):
        with torch.no_grad():
            num_classes = output.shape[1]
            output_seg = output.argmax(1)
            target = target[:, 0]
            axes = tuple(range(1, len(target.shape)))
            tp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
            fp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
            fn_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
            for c in range(1, num_classes):
                tp_hard[:, c - 1] = sum_tensor((output_seg == c).float() * (target == c).float(), axes=axes)
                fp_hard[:, c - 1] = sum_tensor((output_seg == c).float() * (target != c).float(), axes=axes)
                fn_hard[:, c - 1] = sum_tensor((output_seg != c).float() * (target == c).float(), axes=axes)

            # tp_hard, fp_hard, fn_hard = get_tp_fp_fn((output_softmax > (1 / num_classes)).float(), target,
            #                                         axes, None)
            # print_if_rank0("before allgather", tp_hard.shape)
            tp_hard = tp_hard.sum(0, keepdim=False)[None]
            fp_hard = fp_hard.sum(0, keepdim=False)[None]
            fn_hard = fn_hard.sum(0, keepdim=False)[None]

            tp_hard = awesome_allgather_function.apply(tp_hard)
            fp_hard = awesome_allgather_function.apply(fp_hard)
            fn_hard = awesome_allgather_function.apply(fn_hard)

        tp_hard = tp_hard.detach().cpu().numpy().sum(0)
        fp_hard = fp_hard.detach().cpu().numpy().sum(0)
        fn_hard = fn_hard.detach().cpu().numpy().sum(0)
        self.online_eval_foreground_dc.append(list((2 * tp_hard) / (2 * tp_hard + fp_hard + fn_hard + 1e-8)))
        self.online_eval_tp.append(list(tp_hard))
        self.online_eval_fp.append(list(fp_hard))
        self.online_eval_fn.append(list(fn_hard))


    def compute_loss(self, output, target):
        total_loss = None

        # Starting here it gets spicy!
        axes = tuple(range(2, len(output.size())))

        # network does not do softmax. We need to do softmax for dice
        output_softmax = softmax_helper(output)

        # get the tp, fp and fn terms we need
        tp, fp, fn, _ = get_tp_fp_fn_tn(output_softmax, target, axes, mask=None)
        # for dice, compute nominator and denominator so that we have to accumulate only 2 instead of 3 variables
        # do_bg=False in nnUNetTrainer -> [:, 1:]
        nominator = 2 * tp[:, 1:]
        denominator = 2 * tp[:, 1:] + fp[:, 1:] + fn[:, 1:]

        if self.batch_dice:
            # for DDP we need to gather all nominator and denominator terms from all GPUS to do proper batch dice
            nominator = awesome_allgather_function.apply(nominator)
            denominator = awesome_allgather_function.apply(denominator)
            nominator = nominator.sum(0)
            denominator = denominator.sum(0)
        else:
            pass

        ce_loss = self.ce_loss(output, target[:, 0].long())

        # we smooth by 1e-5 to penalize false positives if tp is 0
        dice_loss = (- (nominator + 1e-5) / (denominator + 1e-5)).mean()
        if total_loss is None:
            total_loss = ce_loss + dice_loss
        else:
            total_loss += ce_loss + dice_loss
        return total_loss


class nnUNetTrainerV2_swinunetr_adam_ddp_lr5e4(nnUNetTrainerV2_swinunetr_adam_ddp):
    def __init__(self, plans_file, fold, local_rank, output_folder=None, dataset_directory=None, batch_dice=True,
                 stage=None, unpack_data=True, deterministic=True, distribute_batch_size=False, fp16=False):
        super().__init__(plans_file, fold, local_rank, output_folder=output_folder, dataset_directory=dataset_directory, batch_dice=batch_dice,
                 stage=stage, unpack_data=unpack_data, deterministic=deterministic, distribute_batch_size=distribute_batch_size, fp16=fp16)

        self.initial_lr = 5e-4