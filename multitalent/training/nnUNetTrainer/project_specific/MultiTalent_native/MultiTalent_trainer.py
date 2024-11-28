from __future__ import annotations
import inspect
import multiprocessing
import os
import shutil
import sys
import subprocess
import warnings
from copy import deepcopy
from datetime import datetime
from time import time, sleep
from typing import Union, Tuple, List
from itertools import chain
import numpy as np
import torch
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.nnunet.random_binary_operator import ApplyRandomBinaryOperatorTransform
from batchgeneratorsv2.transforms.nnunet.remove_connected_components import \
    RemoveRandomConnectedComponentFromOneHotEncodingTransform
from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import MoveSegAsOneHotToDataTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform, Convert2DTo3DTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform
from torch import autocast, nn
from torch import distributed as dist
from torch._dynamo import OptimizedModule
from torch.cuda import device_count
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from multitalent.configuration import ANISO_THRESHOLD, default_num_processes
from multitalent.evaluation.evaluate_predictions import compute_metrics_on_folder
from multitalent.inference.export_prediction import export_prediction_from_logits
from multitalent.utilities.MultiTalent.predict_from_raw_data_multitalent import nnUNetPredictor_MT
from multitalent.inference.sliding_window_prediction import compute_gaussian
from multitalent.paths import nnUNet_preprocessed, nnUNet_results
from multitalent.training.data_augmentation.compute_initial_patch_size import get_patch_size
from multitalent.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from multitalent.utilities.MultiTalent.base_data_loader_MT import nnUNetDataLoader3D_MT, nnUNetDataLoader3D_MTall
from multitalent.training.dataloading.nnunet_dataset import nnUNetDataset
from multitalent.training.dataloading.utils import get_case_identifiers, unpack_dataset
from multitalent.utilities.MultiTalent.logger import nnUNetLogger_MT
from multitalent.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from multitalent.training.loss.deep_supervision import DeepSupervisionWrapper
from multitalent.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from multitalent.training.lr_scheduler.polylr import PolyLRScheduler
from multitalent.utilities.MultiTalent.collate_MT import collate_outputs_MT, collate_outputs
from multitalent.utilities.crossval_split import generate_crossval_split
from multitalent.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from multitalent.utilities.file_path_utilities import check_workers_alive_and_busy
from multitalent.utilities.helpers import empty_cache, dummy_context
from multitalent.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels
from multitalent.utilities.plans_handling.plans_handler import PlansManager
from multitalent.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from multitalent.utilities.MultiTalent.get_MT_network_from_plans import get_MT_network_from_plans
from multitalent.utilities.MultiTalent.label_splitting import split_dict_with_background, sample_from_dict


class MultiTalent_trainer(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):

        self.is_ddp = dist.is_available() and dist.is_initialized()
        self.local_rank = 0 if not self.is_ddp else dist.get_rank()

        self.device = device

        # print what device we are using
        if self.is_ddp:  # implicitly it's clear that we use cuda in this case
            print(f"I am local rank {self.local_rank}. {device_count()} GPUs are available. The world size is "
                  f"{dist.get_world_size()}."
                  f"Setting device to {self.device}")
            self.device = torch.device(type='cuda', index=self.local_rank)
        else:
            if self.device.type == 'cuda':
                # we might want to let the user pick this but for now please pick the correct GPU with CUDA_VISIBLE_DEVICES=X
                self.device = torch.device(type='cuda', index=0)
            print(f"Using device: {self.device}")

        # collect all datast ids
        self.all_ids = []
        # for datasets >x classes, we split up the classes
        self.labelmapping = {}
        self.max_classes = 20
        self.num_cases_per_dataset = {}
        for d in dataset_json['dataset_ids']:
            assert type(d) == str
            dataset_tmp = load_json(join(nnUNet_preprocessed, maybe_convert_to_dataset_name(d), 'dataset.json'))
            self.num_cases_per_dataset[d] = dataset_tmp["numTraining"]
            label_tmp = dict(sorted(dataset_tmp['labels'].items(), key=lambda item: item[1]))
            if len(label_tmp.keys()) > self.max_classes:
                splits, _ = split_dict_with_background(label_tmp, self.max_classes)
                for c,v in enumerate(splits):
                    self.all_ids.append(d + '_%d' %c)
                    self.labelmapping[d + '_%d' %c] = []
                    nonzero_values = [value for value in v.values() if value != 0]
                    new_labels = {}
                    for k in v.keys():
                        if k != 'background':
                            if c == 0:
                                self.labelmapping[d + '_%d' %c].append([v[k],v[k]])
                                new_labels[k] = v[k]
                            else:
                                self.labelmapping[d + '_%d' %c].append([v[k], v[k] - (min(nonzero_values)-1)])
                                new_labels[k] = v[k] - (min(nonzero_values) - 1)
                        else:
                            new_labels[k] = v[k]
                        dataset_tmp["labels"] = new_labels

                    save_json(dataset_tmp, join(nnUNet_preprocessed, maybe_convert_to_dataset_name(d), d + '_%d_dataset.json' %c ))
            else:
                self.all_ids.append(d)
                self.labelmapping[d] = None
        # loading and saving this class for continuing from checkpoint should not happen based on pickling. This
        # would also pickle the network etc. Bad, bad. Instead we just reinstantiate and then load the checkpoint we
        # need. So let's save the init args
        self.my_init_kwargs = {}
        for k in inspect.signature(self.__init__).parameters.keys():
            self.my_init_kwargs[k] = locals()[k]

        ###  Saving all the init args into class variables for later access

        #mainplan
        self.plans_manager = PlansManager(plans)

        #individualplans
        self.plans_managers = {}
        for id in self.all_ids:
            raw_id = id.split('_')[0]
            self.plans_managers[id] = PlansManager(load_json(join(nnUNet_preprocessed, maybe_convert_to_dataset_name(raw_id), self.plans_manager.plans_name + '.json')))

        self.configuration_manager = self.plans_manager.get_configuration(configuration)
        self.configuration_name = configuration
        self.preprocessed_dataset_folder_base = join(nnUNet_preprocessed, self.plans_manager.dataset_name) \
            if nnUNet_preprocessed is not None else None
        assert self.check_MT_plans(), "Not all plans configurations are the same!"

        self.configuration_managers = {}
        for id in self.all_ids:
            self.configuration_managers[id] = self.plans_managers[id].get_configuration(configuration)

        #dataset_json for each dataset of MT
        self.dataset_jsons = {}
        for id in self.all_ids:
            if '_' not in id:
                self.dataset_jsons[id] = load_json(join(nnUNet_preprocessed, maybe_convert_to_dataset_name(id), 'dataset.json'))
            else:
                raw_id = id.split('_')[0]
                self.dataset_jsons[id] = load_json(join(nnUNet_preprocessed, maybe_convert_to_dataset_name(raw_id),'%s_dataset.json' %id))


        self.fold = fold
        self.unpack_dataset = unpack_dataset

        ### Setting all the folder names. We need to make sure things don't crash in case we are just running
        # inference and some of the folders may not be defined!
        self.preprocessed_dataset_folder_bases = {}
        for id in self.all_ids:
            self.preprocessed_dataset_folder_bases[id] = join(nnUNet_preprocessed, self.plans_managers[id].dataset_name) \
                if nnUNet_preprocessed is not None else None
        self.output_folder_base = join(nnUNet_results, self.plans_manager.dataset_name,
                                       self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" + configuration) \
            if nnUNet_results is not None else None
        self.output_folder = join(self.output_folder_base, f'fold_{fold}')

        self.preprocessed_dataset_folders = {}
        for id in self.all_ids:
            self.preprocessed_dataset_folders[id] = join(self.preprocessed_dataset_folder_bases[id], self.configuration_managers[id].data_identifier)
        # unlike the previous nnunet folder_with_segs_from_previous_stage is now part of the plans. For now it has to
        # be a different configuration in the same plans
        # IMPORTANT! the mapping must be bijective, so lowres must point to fullres and vice versa (using
        # "previous_stage" and "next_stage"). Otherwise it won't work!
        self.is_cascaded = self.configuration_manager.previous_stage_name is not None
        self.folder_with_segs_from_previous_stage = \
            join(nnUNet_results, self.plans_manager.dataset_name,
                 self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" +
                 self.configuration_manager.previous_stage_name, 'predicted_next_stage', self.configuration_name) \
                if self.is_cascaded else None

        ### Some hyperparameters for you to fiddle with
        self.initial_lr = 1e-3
        self.weight_decay = 3e-5
        self.oversample_foreground_percent = 0.33
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self.num_epochs = 1000
        self.current_epoch = 0
        self.enable_deep_supervision = True

        ### Dealing with labels/regions
        self.label_managers = {}
        for id in self.all_ids :
            self.label_managers[id] = self.plans_managers[id].get_label_manager(self.dataset_jsons[id])

        # we want MT to allow to have different channel inputs
        self.input_channels = {}
        for id in self.all_ids:
            self.input_channels[id] = determine_num_input_channels(self.plans_managers[id], self.configuration_manager,
                                                                   self.dataset_jsons[id])

        # labels can either be a list of int (regular training) or a list of tuples of int (region-based training)
        # needed for predictions. We do sigmoid in case of (overlapping) regions

        self.num_input_channels = None  # -> self.initialize()
        self.network = None  # -> self.build_network_architecture()
        self.optimizer = self.lr_scheduler = None  # -> self.initialize
        self.grad_scaler = GradScaler() if self.device.type == 'cuda' else None
        self.loss = None  # -> self.initialize

        ### Simple logging. Don't take that away from me!
        # initialize log file. This is just our log for the print statements etc. Not to be confused with lightning
        # logging
        timestamp = datetime.now()
        maybe_mkdir_p(self.output_folder)
        self.log_file = join(self.output_folder, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
                             (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
                              timestamp.second))
        self.logger = nnUNetLogger_MT(self.all_ids)

        ### placeholders
        self.dataloader_train = self.dataloader_val = None  # see on_train_start

        ### initializing stuff for remembering things and such
        self._best_ema = None

        ### inference things
        self.inference_allowed_mirroring_axes = None  # this variable is set in
        # self.configure_rotation_dummyDA_mirroring_and_inital_patch_size and will be saved in checkpoints

        ### checkpoint saving stuff
        self.save_every = 50
        self.disable_checkpointing = False

        ## DDP batch size and oversampling can differ between workers and needs adaptation
        # we need to change the batch size in DDP because we don't use any of those distributed samplers
        self._set_batch_size_and_oversample()

        self.was_initialized = False

        self.print_to_log_file("\n#######################################################################\n"
                               "Please cite the following paper when using nnU-Net:\n"
                               "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
                               "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
                               "Nature methods, 18(2), 203-211.\n"
                               "#######################################################################\n",
                               also_print_to_console=True, add_timestamp=False)

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
                        if key.startswith(raw_id+"_"):
                            datset_parts += 1
                    if datset_parts == 0:
                        datset_parts = 1
                    self.sample_probs[id] = props_case_normed[raw_id] * self.num_cases_per_dataset[raw_id] / datset_parts

                for id in self.all_ids:
                    self.print_to_log_file('Prob for Dataset %s is %f' % (id, self.sample_probs[id]))


                #we want MT to allow to have different channel inputs
                input_channels = []
                for id in self.all_ids:
                    input_channels.append(determine_num_input_channels(self.plans_managers[id], self.configuration_manager,
                                                 self.dataset_jsons[id]))
                self.num_input_channels = np.max(input_channels)
                self.print_to_log_file('max_input_channels: ', self.num_input_channels)

                self.all_num_seg_heads = {}
                for id in self.all_ids:
                    #custumn change for the datasets that have a high number of classes to not run in OOM GPU
                    if '_' in id:
                        self.all_num_seg_heads[id] = len(self.labelmapping[id])+1

                    else:
                        self.all_num_seg_heads[id] = self.label_managers[id].num_segmentation_heads


                # critical part for MT, but just need to update self.build_network_architecture

                self.network = self.build_network_architecture(
                    self.configuration_manager.network_arch_class_name,
                    self.configuration_manager.network_arch_init_kwargs,
                    self.configuration_manager.network_arch_init_kwargs_req_import,
                    self.num_input_channels,
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
            for scheme_a in configuration_props['configuration']['normalization_schemes']:
                for scheme_b in cur_props['configuration']['normalization_schemes']:
                    if scheme_a != scheme_b:
                        print(f"Plan configurations of main plan and plan {id} are different in:", "normalization_schemes")
                        return False
        return True


    def _do_i_compile(self):
        # # new default: compile is enabled!
        # # not for MT
        if 'nnUNet_compile' not in os.environ.keys():
           return True
        else:
           return os.environ['nnUNet_compile'].lower() in ('true', '1', 't')

    def _save_debug_information(self):
        # saving some debug information
        if self.local_rank == 0:
            dct = {}
            for k in self.__dir__():
                if not k.startswith("__"):
                    if not callable(getattr(self, k)) or k in ['loss', ]:
                        dct[k] = str(getattr(self, k))
                    elif k in ['network', ]:
                        dct[k] = str(getattr(self, k).__class__.__name__)
                    else:
                        # print(k)
                        pass
                if k in ['dataloader_train', 'dataloader_val']:
                    if hasattr(getattr(self, k), 'generator'):
                        dct[k + '.generator'] = str(getattr(self, k).generator)
                    if hasattr(getattr(self, k), 'num_processes'):
                        dct[k + '.num_processes'] = str(getattr(self, k).num_processes)
                    if hasattr(getattr(self, k), 'transform'):
                        dct[k + '.transform'] = str(getattr(self, k).transform)
            import subprocess
            hostname = subprocess.getoutput(['hostname'])
            dct['hostname'] = hostname
            torch_version = torch.__version__
            if self.device.type == 'cuda':
                gpu_name = torch.cuda.get_device_name()
                dct['gpu_name'] = gpu_name
                cudnn_version = torch.backends.cudnn.version()
            else:
                cudnn_version = 'None'
            dct['device'] = str(self.device)
            dct['torch_version'] = torch_version
            dct['cudnn_version'] = cudnn_version
            save_json(dct, join(self.output_folder, "debug.json"))

    @staticmethod


    def  build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int|dict,
                                   num_output_channels: dict,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        """
        This is where you build the architecture according to the plans. There is no obligation to use
        get_network_from_plans, this is just a utility we use for the nnU-Net default architectures. You can do what
        you want. Even ignore the plans and just return something static (as long as it can process the requested
        patch size)
        but don't bug us with your bugs arising from fiddling with this :-P
        This is the function that is called in inference as well! This is needed so that all network architecture
        variants can be loaded at inference time (inference will use the same nnUNetTrainer that was used for
        training, so if you change the network architecture during training by deriving a new trainer class then
        inference will know about it).

        If you need to know how many segmentation outputs your custom architecture needs to have, use the following snippet:
        > label_manager = plans_manager.get_label_manager(dataset_json)
        > label_manager.num_segmentation_heads
        (why so complicated? -> We can have either classical training (classes) or regions. If we have regions,
        the number of outputs is != the number of classes. Also there is the ignore label for which no output
        should be generated. label_manager takes care of all that for you.)

        """

        # need to be adapted for MT
        network = get_MT_network_from_plans(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision)

        return network


    def _get_deep_supervision_scales(self):
        if self.enable_deep_supervision:
            deep_supervision_scales = list(list(i) for i in 1 / np.cumprod(np.vstack(
                self.configuration_manager.pool_op_kernel_sizes), axis=0))[:-1]
        else:
            deep_supervision_scales = None  # for train and val_transforms
        return deep_supervision_scales

    def _set_batch_size_and_oversample(self):
        if not self.is_ddp:
            # set batch size to what the plan says, leave oversample untouched
            self.batch_size = self.configuration_manager.batch_size

        else:
            # batch size is distributed over DDP workers and we need to change oversample_percent for each worker
            # batch size is distributed over DDP workers and we need to change oversample_percent for each worker

            world_size = dist.get_world_size()
            my_rank = dist.get_rank()

            global_batch_size = self.configuration_manager.batch_size
            assert global_batch_size >= world_size, 'Cannot run DDP if the batch size is smaller than the number of ' \
                                                    'GPUs... Duh.'

            batch_size_per_GPU = [global_batch_size // world_size] * world_size
            batch_size_per_GPU = [batch_size_per_GPU[i] + 1
                                  if (batch_size_per_GPU[i] * world_size + i) < global_batch_size
                                  else batch_size_per_GPU[i]
                                  for i in range(len(batch_size_per_GPU))]
            assert sum(batch_size_per_GPU) == global_batch_size

            sample_id_low = 0 if my_rank == 0 else np.sum(batch_size_per_GPU[:my_rank])
            sample_id_high = np.sum(batch_size_per_GPU[:my_rank + 1])

            # This is how oversampling is determined in DataLoader
            # round(self.batch_size * (1 - self.oversample_foreground_percent))
            # We need to use the same scheme here because an oversample of 0.33 with a batch size of 2 will be rounded
            # to an oversample of 0.5 (1 sample random, one oversampled). This may get lost if we just numerically
            # compute oversample
            oversample = [True if not i < round(global_batch_size * (1 - self.oversample_foreground_percent)) else False
                          for i in range(global_batch_size)]

            if sample_id_high / global_batch_size < (1 - self.oversample_foreground_percent):
                oversample_percent = 0.0
            elif sample_id_low / global_batch_size > (1 - self.oversample_foreground_percent):
                oversample_percent = 1.0
            else:
                oversample_percent = sum(oversample[sample_id_low:sample_id_high]) / batch_size_per_GPU[my_rank]

            print("worker", my_rank, "oversample", oversample_percent)
            print("worker", my_rank, "batch_size", batch_size_per_GPU[my_rank])

            self.batch_size = batch_size_per_GPU[my_rank]
            self.oversample_foreground_percent = oversample_percent

    def _build_loss(self, id):
        if self.is_ddp:
            batchdice = False
        else:
            batchdice = self.configuration_manager.batch_dice
        if self.label_managers[id].has_regions:
            loss = DC_and_BCE_loss({},
                                   {'batch_dice': batchdice,
                                    'do_bg': True, 'smooth': 0, 'ddp': self.is_ddp},
                                   use_ignore_label=self.label_managers[id].ignore_label is not None,
                                   dice_class=MemoryEfficientSoftDiceLoss)
        else:
            loss = DC_and_CE_loss({'batch_dice': batchdice,
                                   'smooth': 0, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                                  ignore_label=self.label_managers[id].ignore_label, dice_class=MemoryEfficientSoftDiceLoss)

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                # Anywho, the simple fix is to set a very low weight to this.
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)

        return loss



    def do_split_individual(self, id:str):
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

        path_tr_keys = []
        path_val_keys = []

        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            case_identifiers = get_case_identifiers(self.preprocessed_dataset_folders[id])
            tr_keys = case_identifiers
            val_keys = tr_keys
        else:
            splits_file = join(self.preprocessed_dataset_folder_bases[id], "splits_final.json")
            dataset = nnUNetDataset(self.preprocessed_dataset_folders[id], case_identifiers=None,
                                    num_images_properties_loading_threshold=0)
            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file(f"Creating new 5-fold cross-validation split for {id} ...")
                all_keys_sorted = list(np.sort(list(dataset.keys())))
                splits = generate_crossval_split(all_keys_sorted, seed=12345, n_splits=5)
                save_json(splits, splits_file)

            else:
                self.print_to_log_file(f"Using splits from existing split file for dataset {id}:", splits_file)
                splits = load_json(splits_file)
                self.print_to_log_file(f"The split file of {id} contains {len(splits)} splits.")

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
                keys = np.sort(list(dataset.keys()))
                idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
                idx_val = [i for i in range(len(keys)) if i not in idx_tr]
                tr_keys = [keys[i] for i in idx_tr]
                val_keys = [keys[i] for i in idx_val]
                self.print_to_log_file("This random 80:20 split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            if any([i in val_keys for i in tr_keys]):
                self.print_to_log_file('WARNING: Some validation cases are also in the training set. Please check the '
                                       'splits.json or ignore if this is intentional.')

        path_list = self.preprocessed_dataset_folders[id].split('/')
        for case in tr_keys:
            path_tr_keys.append(join(path_list[-2], path_list[-1], case))
        for case in val_keys:
            path_val_keys.append(join(join(path_list[-2], path_list[-1], case)))

        return path_tr_keys, path_val_keys

    def get_tr_and_val_datasets(self):
        # create dataset split

        # instead of only the case identifier, we want also the path from nnunet preprocessed
        tr_keys = []
        val_keys = []
        for id in self.all_ids:
            if id.endswith('_0'):
                tr_keys_tmp, val_keys_tmp = self.do_split_individual(id)
                tr_keys += tr_keys_tmp
                val_keys += val_keys_tmp
            if "_" not in id:
                tr_keys_tmp, val_keys_tmp = self.do_split_individual(id)
                tr_keys += tr_keys_tmp
                val_keys += val_keys_tmp


        # load the datasets for training and validation. Note that we always draw random samples so we really don't
        # care about distributing training cases across GPUs.
        dataset_tr = nnUNetDataset(nnUNet_preprocessed, tr_keys,
                                   folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                   num_images_properties_loading_threshold=0)
        dataset_val = nnUNetDataset(nnUNet_preprocessed, val_keys,
                                    folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                    num_images_properties_loading_threshold=0)
        return dataset_tr, dataset_val

    def get_dataloaders(self):
        patch_size = self.configuration_manager.patch_size

        # needed for deep supervision: how much do we need to downscale the segmentation targets for the different
        # outputs?

        deep_supervision_scales = self._get_deep_supervision_scales()

        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        # training pipeline
        tr_transforms = {}
        val_transforms = {}
        for id in self.all_ids:
            tr_transforms[id] = self.get_training_transforms(
                patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
                use_mask_for_norm=self.configuration_managers[id].use_mask_for_norm,
                is_cascaded=self.is_cascaded, foreground_labels=self.label_managers[id].foreground_labels,
                regions= self.label_managers[id].foreground_regions if self.label_managers[id].has_regions else None,
                ignore_label= self.label_managers[id].ignore_label)

            # validation pipeline
            val_transforms[id] = self.get_validation_transforms(deep_supervision_scales,
                                                            is_cascaded=self.is_cascaded,
                                                            foreground_labels=self.label_managers[id].foreground_labels,
                                                            regions=self.label_managers[id].foreground_regions if self.label_managers[id].has_regions else None,
                                                            ignore_label=self.label_managers[id].ignore_label)


        dataset_tr, dataset_val = self.get_tr_and_val_datasets()
        keys = list(dataset_tr.keys())
        dataset_dentifiers = list(np.unique([i.split('/')[0][7:10] for i in keys]))
        num_cases_per_dataset = [len([i for i in keys if i.startswith('Dataset' +j + '_')]) for j in dataset_dentifiers]
        probabilities = np.array(
            [1 / (num_cases_per_dataset[dataset_dentifiers.index(i.split('/')[0][7:10])] ** 0.5) for i in keys])
        probabilities = probabilities / sum(probabilities)
        self.dataset_prob = {}
        self.print_to_log_file('cases per datasset train:\n', list(zip(dataset_dentifiers, num_cases_per_dataset)))
        self.print_to_log_file('probabilities per dataset:')
        for d in dataset_dentifiers:
            dataset_keys = [i for i in keys if i.startswith('Dataset' + d + '_')]
            p_per_case = probabilities[keys.index(dataset_keys[0])]
            p_per_dataset = p_per_case * len(dataset_keys)
            self.print_to_log_file(d, p_per_case, p_per_dataset)
            self.dataset_prob[d] = p_per_dataset

        keys_val = list(dataset_val.keys())
        dataset_dentifiers_val = list(np.unique([i.split('_')[0][7:10] for i in keys_val]))
        num_cases_per_dataset = [len([i for i in keys_val if i.startswith('Dataset' +j + '_')]) for j in dataset_dentifiers_val]
        probabilities_val = np.array(
            [1 / (num_cases_per_dataset[dataset_dentifiers_val.index(i.split('/')[0][7:10])] ** 0.5) for i in keys_val])
        probabilities_val = probabilities_val / sum(probabilities_val)



        dl_tr = nnUNetDataLoader3D_MTall(dataset_tr, 1,
                                      initial_patch_size,
                                      self.configuration_manager.patch_size,
                                      self.label_managers,
                                      oversample_foreground_percent=self.oversample_foreground_percent,
                                      sampling_probabilities=probabilities, pad_sides=None,transforms=tr_transforms,
                                      labelmapping=self.labelmapping, input_channels = self.input_channels)
        dl_val = nnUNetDataLoader3D_MTall(dataset_val, 1,
                                       self.configuration_manager.patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_managers,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=probabilities_val, pad_sides=None, transforms=val_transforms,
                                       labelmapping=self.labelmapping, input_channels = self.input_channels)


        allowed_num_processes = get_allowed_n_proc_DA()


        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(data_loader=dl_tr, transform=None,
                                                        num_processes=allowed_num_processes,
                                                        num_cached=max(12*self.batch_size, allowed_num_processes*self.batch_size), seeds=None,
                                                        pin_memory=self.device.type == 'cuda', wait_time=0.02)
            mt_gen_val = NonDetMultiThreadedAugmenter(data_loader=dl_val,
                                                      transform=None, num_processes=max(1, allowed_num_processes // 2),
                                                      num_cached=max(6*self.batch_size, (allowed_num_processes*self.batch_size) // 4), seeds=None,
                                                      pin_memory=self.device.type == 'cuda',
                                                      wait_time=0.02)
        # # let's get this party started
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)

        return mt_gen_train, mt_gen_val

    @staticmethod
    def get_training_transforms(
        patch_size: Union[np.ndarray, Tuple[int]],
        rotation_for_DA: RandomScalar,
        deep_supervision_scales: Union[List, Tuple, None],
        mirror_axes: Tuple[int, ...],
        do_dummy_2d_data_aug: bool,
        use_mask_for_norm: List[bool] = None,
        is_cascaded: bool = False,
        foreground_labels: Union[Tuple[int, ...], List[int]] = None,
        regions: List[Union[List[int], Tuple[int, ...], int]] = None,
        ignore_label: int = None,
    ) -> BasicTransform:
        transforms = []
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None

        transforms.append(
            SpatialTransform(
                patch_size_spatial, patch_center_dist_from_border=0, random_crop=False, p_elastic_deform=0,
                p_rotation=0.2,
                rotation=rotation_for_DA, p_scaling=0.2, scaling=(0.7, 1.4), p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False  # , mode_seg='nearest'
            )
        )


        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())

        transforms.append(RandomTransform(
            GaussianNoiseTransform(
                noise_variance=(0, 0.1),
                p_per_channel=1,
                synchronize_channels=True
            ), apply_probability=0.1
        ))
        transforms.append(RandomTransform(
            GaussianBlurTransform(
                blur_sigma=(0.5, 1.),
                synchronize_channels=False,
                synchronize_axes=False,
                p_per_channel=0.5, benchmark=True
            ), apply_probability=0.2
        ))
        transforms.append(RandomTransform(
            MultiplicativeBrightnessTransform(
                multiplier_range=BGContrast((0.75, 1.25)),
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            ContrastTransform(
                contrast_range=BGContrast((0.75, 1.25)),
                preserve_range=True,
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            SimulateLowResolutionTransform(
                scale=(0.5, 1),
                synchronize_channels=False,
                synchronize_axes=True,
                ignore_axes=ignore_axes,
                allowed_channels=None,
                p_per_channel=0.5
            ), apply_probability=0.25
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=1,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.1
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=0,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.3
        ))

        if mirror_axes is not None and len(mirror_axes) > 0:
                transforms.append(
                    MirrorTransform(
                        allowed_axes=mirror_axes
                    )
                )


        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms.append(MaskImageTransform(
                apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                channel_idx_in_seg=0,
                set_outside_to=0,
            ))

        transforms.append(
            RemoveLabelTansform(-1, 0)
        )

        if is_cascaded:
            assert foreground_labels is not None, 'We need foreground_labels for cascade augmentations'
            transforms.append(
                MoveSegAsOneHotToDataTransform(
                    source_channel_idx=1,
                    all_labels=foreground_labels,
                    remove_channel_from_source=True
                )
            )
            transforms.append(
                RandomTransform(
                    ApplyRandomBinaryOperatorTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        strel_size=(1, 8),
                        p_per_label=1
                    ), apply_probability=0.4
                )
            )
            transforms.append(
                RandomTransform(
                    RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        fill_with_other_class_p=0,
                        dont_do_if_covers_more_than_x_percent=0.15,
                        p_per_label=1
                    ), apply_probability=0.2
                )
            )


        if regions is not None:
            # the ignore label must also be converted
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                    channel_in_seg=0
                )
            )


        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))

        return ComposeTransforms(transforms)


    @staticmethod
    def get_validation_transforms(
        deep_supervision_scales: Union[List, Tuple, None],
        is_cascaded: bool = False,
        foreground_labels: Union[Tuple[int, ...], List[int]] = None,
        regions: List[Union[List[int], Tuple[int, ...], int]] = None,
        ignore_label: int = None,
        ) -> BasicTransform:
        transforms = []
        transforms.append(
            RemoveLabelTansform(-1, 0)
        )

        if is_cascaded:
            transforms.append(
                MoveSegAsOneHotToDataTransform(
                    source_channel_idx=1,
                    all_labels=foreground_labels,
                    remove_channel_from_source=True
                )
            )


        if regions is not None:
            # the ignore label must also be converted
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                    channel_in_seg=0
                )
            )


        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))
        return ComposeTransforms(transforms)

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

        mod.decoder.deep_supervision = enabled

    def on_train_start(self):
        self.dataloader_train, self.dataloader_val = self.get_dataloaders()

        if not self.was_initialized:
            self.initialize()

        maybe_mkdir_p(self.output_folder)

        # make sure deep supervision is on in the network
        self.set_deep_supervision_enabled(self.enable_deep_supervision)

        self.print_plans()
        empty_cache(self.device)

        # maybe unpack
        if self.unpack_dataset and self.local_rank == 0:
            self.print_to_log_file('unpacking dataset...')
            for id in self.all_ids:
                unpack_dataset(self.preprocessed_dataset_folders[id], unpack_segmentation=False, overwrite_existing=False,
                               num_processes=max(1, round(get_allowed_n_proc_DA() // 2)))
            self.print_to_log_file('unpacking done...')

        if self.is_ddp:
            dist.barrier()

        # dataloaders must be instantiated here because they need access to the training data which may not be present
        # when doing inference
        if self.local_rank == 0:
            # copy plans and dataset.json so that they can be used for restoring everything we need for inference
            # just to be save, we save all plans, datasetjsons and fingerprints from all datasets
            for id in self.all_ids:
                save_json(self.plans_managers[id].plans, join(self.output_folder_base,str(id)+'_plans.json'), sort_keys=False)
                save_json(self.dataset_jsons[id], join(self.output_folder_base, str(id)+'_dataset.json'), sort_keys=False)

            # we don't really need the fingerprint but its still handy to have it with the others
                shutil.copy(join(self.preprocessed_dataset_folder_bases[id], 'dataset_fingerprint.json'),
                            join(self.output_folder_base, str(id) + '_dataset_fingerprint.json'))

        # produces a pdf in output folder
        self.plot_network_architecture()

        self._save_debug_information()

        # print(f"batch size: {self.batch_size}")
        # print(f"oversample: {self.oversample_foreground_percent}")

    def on_train_end(self):
        # dirty hack because on_epoch_end increments the epoch counter and this is executed afterwards.
        # This will lead to the wrong current epoch to be stored
        self.current_epoch -= 1
        self.save_checkpoint(join(self.output_folder, "checkpoint_final.pth"))
        self.current_epoch += 1

        # now we can delete latest
        if self.local_rank == 0 and isfile(join(self.output_folder, "checkpoint_latest.pth")):
            os.remove(join(self.output_folder, "checkpoint_latest.pth"))

        # shut down dataloaders
        old_stdout = sys.stdout
        with open(os.devnull, 'w') as f:
            sys.stdout = f
            if self.dataloader_train is not None:
                self.dataloader_train._finish()
            if self.dataloader_val is not None:
                self.dataloader_val._finish()
            sys.stdout = old_stdout

        empty_cache(self.device)
        self.print_to_log_file("Training done.")

    def on_train_epoch_start(self):
        self.network.train()
        self.lr_scheduler.step(self.current_epoch)
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch}')
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}")
        # lrs are the same for all workers so we don't need to gather them in case of DDP training
        self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)

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

    def on_train_epoch_end(self, train_outputs: List[dict]):
        outputs = collate_outputs(train_outputs)

        if self.is_ddp:
            losses_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_tr, outputs['loss'])
            loss_here = np.vstack(losses_tr).mean()
        else:
            loss_here = np.mean(outputs['loss'])

        self.logger.log('train_losses', loss_here, self.current_epoch)

    def on_validation_epoch_start(self):
        self.network.eval()

    def validation_step(self, batch: list[dict], ids: list) -> dict:
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

        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output_net = self.network(data, ids)
            del data
            for c,id in enumerate(ids):
                if c == 0:
                    l = self.loss[id](output_net[c], b_target[c])
                else:
                    l += self.loss[id](output_net[c], b_target[c])

        tp_hard_all = {}
        fp_hard_all = {}
        fn_hard_all = {}

        # we only need the output with the highest output resolution (if DS enabled)
        for c, id in enumerate(ids):
            if self.enable_deep_supervision:
                output = output_net[c][0]
                target = b_target[c][0]
            else:
                output = output_net[c]
                target = b_target[c], dim=1
                # the following is needed for online evaluation. Fake dice (green line)
            axes = [0] + list(range(2, output.ndim))

            if self.label_managers[id].has_regions:
                predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
            else:
                # no need for softmax
                output_seg = output.argmax(1)[:, None]
                predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
                predicted_segmentation_onehot.scatter_(1, output_seg, 1)
                del output_seg

            if self.label_managers[id].has_ignore_label:
                if not self.label_managers[id].has_regions:
                    mask = (target != self.label_managers[id].ignore_label).float()
                    # CAREFUL that you don't rely on target after this line!
                    target[target == self.label_managers[id].ignore_label] = 0
                else:
                    mask = 1 - target[:, -1:]
                    # CAREFUL that you don't rely on target after this line!
                    target = target[:, :-1]
            else:
                mask = None

            tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

            tp_hard = tp.detach().cpu().numpy()
            fp_hard = fp.detach().cpu().numpy()
            fn_hard = fn.detach().cpu().numpy()
            if not self.label_managers[id].has_regions:
                # if we train with regions all segmentation heads predict some kind of foreground. In conventional
                # (softmax training) there needs tobe one output for the background. We are not interested in the
                # background Dice
                # [1:] in order to remove background
                if id in tp_hard_all.keys():
                    tp_hard_all[id] += tp_hard[1:]
                    fp_hard_all[id] += fp_hard[1:]
                    fn_hard_all[id] += fn_hard[1:]
                else:
                    tp_hard_all[id] = tp_hard[1:]
                    fp_hard_all[id] = fp_hard[1:]
                    fn_hard_all[id] = fn_hard[1:]
            else:
                if id in tp_hard_all.keys():
                    tp_hard_all[id] += tp_hard
                    fp_hard_all[id] += fp_hard
                    fn_hard_all[id] += fn_hard
                else:
                    tp_hard_all[id] = tp_hard
                    fp_hard_all[id] = fp_hard
                    fn_hard_all[id] = fn_hard

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard_all, 'fp_hard': fp_hard_all, 'fn_hard': fn_hard_all}

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs_MT(val_outputs, self.all_ids)
        all_classes_dice = []
        for id in self.all_ids:
            tp = np.sum(outputs_collated[id]['tp_hard'], 0)
            fp = np.sum(outputs_collated[id]['fp_hard'], 0)
            fn = np.sum(outputs_collated[id]['fn_hard'], 0)

            if not isinstance(fp, (np.ndarray,list)):
                fp = np.array([fp]* (self.plans_managers[id].get_label_manager(self.dataset_jsons[id]).num_segmentation_heads -1))
            if not isinstance(tp, (np.ndarray,list)):
                tp = np.array([tp]* (self.plans_managers[id].get_label_manager(self.dataset_jsons[id]).num_segmentation_heads -1))
            if not isinstance(fn, (np.ndarray,list)):
                fn = np.array([fn]* (self.plans_managers[id].get_label_manager(self.dataset_jsons[id]).num_segmentation_heads -1))

            try:
                if self.is_ddp:
                    world_size = dist.get_world_size()

                    tps = [None for _ in range(world_size)]
                    dist.all_gather_object(tps, tp)
                    tp = np.vstack([i[None] for i in tps]).sum(0)

                    fps = [None for _ in range(world_size)]
                    dist.all_gather_object(fps, fp)
                    fp = np.vstack([i[None] for i in fps]).sum(0)

                    fns = [None for _ in range(world_size)]
                    dist.all_gather_object(fns, fn)
                    fn = np.vstack([i[None] for i in fns]).sum(0)
                global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)]]
            except:
                global_dc_per_class = [np.nan]*(len(tp))
            if self.current_epoch%10 == 0:
                self.print_to_log_file(id, global_dc_per_class)
                self.logger.log('dice_per_class_or_region_' + id , global_dc_per_class, self.current_epoch)
            all_classes_dice.append(global_dc_per_class)

        if self.is_ddp:
            losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(losses_val, outputs_collated['loss'])
            loss_here = np.nanmean(np.vstack(losses_val))
        else:
            loss_here = np.nanmean(outputs_collated['loss'])
        if self.current_epoch%10 == 0:
            self.logger.log('val_losses', loss_here, self.current_epoch)
        mean_fg_dice = np.nanmean(list(chain(*all_classes_dice)))
        if self.current_epoch % 10 == 0:
            self.logger.log('mean_fg_dice', mean_fg_dice, self.current_epoch)

    def on_epoch_start(self):
        self.logger.log('epoch_start_timestamps', time(), self.current_epoch)

    def on_epoch_end(self):
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
        if self.current_epoch % 10 == 0 or self.current_epoch == 0:
            for id in self.all_ids:
                self.print_to_log_file('Pseudo dice for dataset ' + id , [np.round(i, decimals=4) for i in
                                                       self.logger.my_fantastic_logging['dice_per_class_or_region_'+id][-1]])
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

        # handling periodic checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
        if self._best_ema is None or self.logger.my_fantastic_logging['ema_fg_dice'][-1] > self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
            self.print_to_log_file(f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1

    def save_checkpoint(self, filename: str) -> None:
        if self.local_rank == 0:
            if not self.disable_checkpointing:
                if self.is_ddp:
                    mod = self.network.module
                else:
                    mod = self.network
                if isinstance(mod, OptimizedModule):
                    mod = mod._orig_mod

                checkpoint = {
                    'network_weights': mod.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'grad_scaler_state': self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
                    'logging': self.logger.get_checkpoint(),
                    '_best_ema': self._best_ema,
                    'current_epoch': self.current_epoch + 1,
                    'init_args': self.my_init_kwargs,
                    'trainer_name': self.__class__.__name__,
                    'inference_allowed_mirroring_axes': self.inference_allowed_mirroring_axes,
                }
                torch.save(checkpoint, filename)
            else:
                self.print_to_log_file('No checkpoint written, checkpointing is disabled')

    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        if not self.was_initialized:
            self.initialize()

        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint, map_location=self.device)
        # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        new_state_dict = {}
        for k, value in checkpoint['network_weights'].items():
            key = k
            if key not in self.network.state_dict().keys() and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value

        self.my_init_kwargs = checkpoint['init_args']
        self.current_epoch = checkpoint['current_epoch']
        self.logger.load_checkpoint(checkpoint['logging'])
        self._best_ema = checkpoint['_best_ema']
        self.inference_allowed_mirroring_axes = checkpoint[
            'inference_allowed_mirroring_axes'] if 'inference_allowed_mirroring_axes' in checkpoint.keys() else self.inference_allowed_mirroring_axes

        # messing with state dict naming schemes. Facepalm.
        if self.is_ddp:
            if isinstance(self.network.module, OptimizedModule):
                self.network.module._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.module.load_state_dict(new_state_dict)
        else:
            if isinstance(self.network, OptimizedModule):
                self.network._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.load_state_dict(new_state_dict)
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if self.grad_scaler is not None:
            if checkpoint['grad_scaler_state'] is not None:
                self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])

    def perform_actual_validation(self, save_probabilities: bool = False):
        self.set_deep_supervision_enabled(False)
        self.network.eval()
        target_json = load_json(join(self.preprocessed_dataset_folder_base, 'val_datasets.json'))
        for id in self.all_ids:
            #custumn
            if id in target_json['dataset_ids']:
                predictor = nnUNetPredictor_MT(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                            perform_everything_on_device=True, device=self.device, verbose=False,
                                            verbose_preprocessing=False, allow_tqdm=False, target_dataset_id=id)
                predictor.manual_initialization(self.network, self.plans_managers[id], self.configuration_managers[id], None,
                                                self.dataset_jsons[id], self.__class__.__name__,
                                                self.inference_allowed_mirroring_axes)

                with multiprocessing.get_context("spawn").Pool(default_num_processes) as segmentation_export_pool:
                    worker_list = [i for i in segmentation_export_pool._pool]
                    validation_output_folder = join(self.output_folder, 'validation_'+id)
                    maybe_mkdir_p(validation_output_folder)

                    # we cannot use self.get_tr_and_val_datasets() here because we might be DDP and then we have to distribute
                    # the validation keys across the workers.
                    _, val_keys = self.do_split_individual(id)

                    if self.is_ddp:
                        val_keys = val_keys[self.local_rank:: dist.get_world_size()]

                    dataset_val = nnUNetDataset(nnUNet_preprocessed, val_keys,
                                                folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                                num_images_properties_loading_threshold=0)

                    next_stages = self.configuration_manager.next_stage_names

                    if next_stages is not None:
                        _ = [maybe_mkdir_p(join(self.output_folder_base, 'predicted_next_stage', n)) for n in next_stages]

                    results = []

                    for k in dataset_val.keys():
                        proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                         allowed_num_queued=2)
                        while not proceed:
                            sleep(0.1)
                            proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                             allowed_num_queued=2)

                        self.print_to_log_file(f"predicting {k}")
                        data, seg, properties = dataset_val.load_case(k)
                        data_shape = data.shape


                        if self.is_cascaded:
                            data = np.vstack((data, convert_labelmap_to_one_hot(seg[-1], self.label_manager.foreground_labels,
                                                                                output_dtype=data.dtype)))
                        with warnings.catch_warnings():
                            # ignore 'The given NumPy array is not writable' warning
                            warnings.simplefilter("ignore")
                            data = torch.from_numpy(data)

                        output_filename_truncated = join(validation_output_folder, k.split('/')[-1])
                        self.print_to_log_file(output_filename_truncated, 'out')
                        if not isfile(output_filename_truncated +'.nii.gz'):

                            try:
                                prediction = predictor.predict_sliding_window_return_logits(data)
                            except RuntimeError:
                                predictor.perform_everything_on_gpu = False
                                prediction = predictor.predict_sliding_window_return_logits(data)
                                predictor.perform_everything_on_gpu = True

                            prediction = prediction.cpu()

                            # this needs to go into background processes
                            results.append(
                                segmentation_export_pool.starmap_async(
                                    export_prediction_from_logits, (
                                        (prediction, properties, self.configuration_managers[id], self.plans_managers[id],
                                         self.dataset_jsons[id], output_filename_truncated, save_probabilities),
                                    )
                                )
                            )

                    _ = [r.get() for r in results]

                if self.is_ddp:
                    dist.barrier()
                if self.local_rank == 0:
                    metrics = compute_metrics_on_folder(join(self.preprocessed_dataset_folder_bases[id], 'gt_segmentations'),
                                                        validation_output_folder,
                                                        join(validation_output_folder, 'summary.json'),
                                                        self.plans_managers[id].image_reader_writer_class(),
                                                        self.dataset_jsons[id]["file_ending"],
                                                        self.label_managers[id].foreground_regions if self.label_managers[id].has_regions else
                                                        self.label_managers[id].foreground_labels,
                                                        self.label_managers[id].ignore_label, chill=True)
                    self.print_to_log_file("Validation complete", also_print_to_console=True)
                    self.print_to_log_file("Mean Validation Dice: ", (metrics['foreground_mean']["Dice"]), also_print_to_console=True)

        self.set_deep_supervision_enabled(True)
        compute_gaussian.cache_clear()

    def run_training(self):
        self.on_train_start()

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []

            for batch_id in range(self.num_iterations_per_epoch):
                next_data = []
                ids = []
                for i in range(self.batch_size):
                    d = next(self.dataloader_train)
                    if d['keys'][0] == "Dataset201_MS_Flair/nnUNetPlans1x1x1_znorm_3d_fullres/Patient-44":
                        d = next(self.dataloader_train)
                    next_data.append(d)
                    ids.append(next_data[i]['id'])
                train_outputs.append(self.train_step(next_data, ids))
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    next_data = []
                    ids = []
                    for i in range(self.batch_size):
                        next_data.append(next(self.dataloader_val))
                        ids.append(next_data[i]['id'])
                    val_outputs.append(self.validation_step(next_data, ids))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()

        self.on_train_end()
class MultiTalent_trainer_1ep(MultiTalent_trainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        self.num_epochs = 1
class MultiTalent_trainer_4000ep(MultiTalent_trainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        self.num_epochs = 4000

