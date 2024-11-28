import inspect
from multitalent.utilities.helpers import empty_cache
from torch import distributed as dist
from torch.cuda import device_count
from torch.cuda.amp import GradScaler
import os
from multitalent.training.dataloading.utils import get_case_identifiers, unpack_dataset
from multitalent.utilities.plans_handling.plans_handler import PlansManager

from time import time
from typing import List, Union, Tuple
import numpy as np
import torch
from torch._dynamo import OptimizedModule
from multitalent.training.dataloading.utils import get_case_identifiers, unpack_dataset
from multitalent.utilities.crossval_split import generate_crossval_split
from multitalent.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from multitalent.training.loss.deep_supervision import DeepSupervisionWrapper
from multitalent.training.loss.dice import  MemoryEfficientSoftDiceLoss
from multitalent.training.lr_scheduler.polylr import PolyLRScheduler_offset, Lin_incr_LRScheduler, Lin_incr_offset_LRScheduler, PolyLRScheduler
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from multitalent.utilities.label_handling.label_handling import determine_num_input_channels
from torch.nn.parallel import DistributedDataParallel as DDP
from multitalent.training.dataloading.nnunet_dataset import nnUNetDataset
from multitalent.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from multitalent.training.nnUNetTrainer.variants.loss.nnUNetTrainerDiceLoss import nnUNetTrainerDiceCELoss_noSmooth
from datetime import datetime
from multitalent.training.logging.nnunet_logger import nnUNetLogger
from multitalent.paths import nnUNet_preprocessed, nnUNet_results

class nnUNetTrainer_20img(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        self.num_train_cases = 20
    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.json file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.json file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            case_identifiers = get_case_identifiers(self.preprocessed_dataset_folder)
            tr_keys = case_identifiers
            val_keys = tr_keys
        else:
            splits_file = join(self.preprocessed_dataset_folder_base, "splits_final.json")
            print(self.preprocessed_dataset_folder)
            dataset = nnUNetDataset(self.preprocessed_dataset_folder, case_identifiers=None,
                                    num_images_properties_loading_threshold=0,
                                    folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                all_keys_sorted = list(np.sort(list(dataset.keys())))

                splits = generate_crossval_split(all_keys_sorted, seed=12345, n_splits=5)
                save_json(splits, splits_file)

            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_json(splits_file)
                self.print_to_log_file(f"The split file contains {len(splits)} splits.")

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


        rnd = np.random.RandomState(seed=12345 + self.fold)
        idx_subsample = rnd.choice(len(tr_keys),  self.num_train_cases, replace=False)
        tr_keys = [tr_keys[i] for i in idx_subsample]
        self.print_to_log_file('We train only with the following %d cases' % self.num_train_cases)

        return tr_keys, val_keys
class nnUNetTrainer_pretrained(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        # From https://grugbrain.dev/. Worth a read ya big brains ;-)

        # apex predator of grug is complexity
        # complexity bad
        # say again:
        # complexity very bad
        # you say now:
        # complexity very, very bad
        # given choice between complexity or one on one against t-rex, grug take t-rex: at least grug see t-rex
        # complexity is spirit demon that enter codebase through well-meaning but ultimately very clubbable non grug-brain developers and project managers who not fear complexity spirit demon or even know about sometime
        # one day code base understandable and grug can get work done, everything good!
        # next day impossible: complexity demon spirit has entered code and very dangerous situation!

        # OK OK I am guilty. But I tried.
        # https://www.osnews.com/images/comics/wtfm.jpg
        # https://i.pinimg.com/originals/26/b2/50/26b250a738ea4abc7a5af4d42ad93af0.jpg

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

        # loading and saving this class for continuing from checkpoint should not happen based on pickling. This
        # would also pickle the network etc. Bad, bad. Instead we just reinstantiate and then load the checkpoint we
        # need. So let's save the init args
        self.my_init_kwargs = {}
        for k in inspect.signature(self.__init__).parameters.keys():
            self.my_init_kwargs[k] = locals()[k]

        ###  Saving all the init args into class variables for later access
        self.plans_manager = PlansManager(plans)
        self.configuration_manager = self.plans_manager.get_configuration(configuration)
        self.configuration_name = configuration
        self.dataset_json = dataset_json
        self.fold = fold
        self.unpack_dataset = unpack_dataset

        ### Setting all the folder names. We need to make sure things don't crash in case we are just running
        # inference and some of the folders may not be defined!
        self.preprocessed_dataset_folder_base = join(nnUNet_preprocessed, self.plans_manager.dataset_name) \
            if nnUNet_preprocessed is not None else None
        self.output_folder_base = join(nnUNet_results, self.plans_manager.dataset_name,
                                       self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" + configuration) \
            if nnUNet_results is not None else None
        if 'pretrain_name' in plans.keys():
            pretrain_name = plans['pretrain_name']
        else:
            pretrain_name= None
        if pretrain_name is not None:
            if 'dublicate' in plans.keys():
                dublicate = '_dublicate_' + plans['dublicate']
            else:
                dublicate = ''
            self.output_folder_base = join(nnUNet_results, self.plans_manager.dataset_name, 'pretrained', pretrain_name + dublicate,
                                           self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" + configuration) \
                if nnUNet_results is not None else None
            self.output_folder = join(self.output_folder_base, f'fold_{fold}')

        else:
            raise RuntimeError('No pretraining weights provided. Makes no sense to use this trainier')

        self.preprocessed_dataset_folder = join(self.preprocessed_dataset_folder_base,
                                                self.configuration_manager.data_identifier)
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
        self.initial_lr = 1e-2
        self.weight_decay = 3e-5
        self.oversample_foreground_percent = 0.33
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self.num_epochs = 1000
        self.current_epoch = 0
        self.enable_deep_supervision = True


        ### Dealing with labels/regions
        self.label_manager = self.plans_manager.get_label_manager(dataset_json)
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
        self.logger = nnUNetLogger()

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
class nnUNetTrainer_pretrained_20img(nnUNetTrainer_pretrained):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        self.num_train_cases = 20
    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.json file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.json file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            case_identifiers = get_case_identifiers(self.preprocessed_dataset_folder)
            tr_keys = case_identifiers
            val_keys = tr_keys
        else:
            splits_file = join(self.preprocessed_dataset_folder_base, "splits_final.json")
            print(self.preprocessed_dataset_folder)
            dataset = nnUNetDataset(self.preprocessed_dataset_folder, case_identifiers=None,
                                    num_images_properties_loading_threshold=0,
                                    folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                all_keys_sorted = list(np.sort(list(dataset.keys())))

                splits = generate_crossval_split(all_keys_sorted, seed=12345, n_splits=5)
                save_json(splits, splits_file)

            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_json(splits_file)
                self.print_to_log_file(f"The split file contains {len(splits)} splits.")

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


        rnd = np.random.RandomState(seed=12345 + self.fold)
        idx_subsample = rnd.choice(len(tr_keys),  self.num_train_cases, replace=False)
        tr_keys = [tr_keys[i] for i in idx_subsample]
        self.print_to_log_file('We train only with the following %d cases' % self.num_train_cases)

        return tr_keys, val_keys
class nnUNetTrainer_pretrained_nosmooth(nnUNetTrainerDiceCELoss_noSmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        # From https://grugbrain.dev/. Worth a read ya big brains ;-)

        # apex predator of grug is complexity
        # complexity bad
        # say again:
        # complexity very bad
        # you say now:
        # complexity very, very bad
        # given choice between complexity or one on one against t-rex, grug take t-rex: at least grug see t-rex
        # complexity is spirit demon that enter codebase through well-meaning but ultimately very clubbable non grug-brain developers and project managers who not fear complexity spirit demon or even know about sometime
        # one day code base understandable and grug can get work done, everything good!
        # next day impossible: complexity demon spirit has entered code and very dangerous situation!

        # OK OK I am guilty. But I tried.
        # https://www.osnews.com/images/comics/wtfm.jpg
        # https://i.pinimg.com/originals/26/b2/50/26b250a738ea4abc7a5af4d42ad93af0.jpg

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

        # loading and saving this class for continuing from checkpoint should not happen based on pickling. This
        # would also pickle the network etc. Bad, bad. Instead we just reinstantiate and then load the checkpoint we
        # need. So let's save the init args
        self.my_init_kwargs = {}
        for k in inspect.signature(self.__init__).parameters.keys():
            self.my_init_kwargs[k] = locals()[k]

        ###  Saving all the init args into class variables for later access
        self.plans_manager = PlansManager(plans)
        self.configuration_manager = self.plans_manager.get_configuration(configuration)
        self.configuration_name = configuration
        self.dataset_json = dataset_json
        self.fold = fold
        self.unpack_dataset = unpack_dataset

        ### Setting all the folder names. We need to make sure things don't crash in case we are just running
        # inference and some of the folders may not be defined!
        self.preprocessed_dataset_folder_base = join(nnUNet_preprocessed, self.plans_manager.dataset_name) \
            if nnUNet_preprocessed is not None else None
        self.output_folder_base = join(nnUNet_results, self.plans_manager.dataset_name,
                                       self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" + configuration) \
            if nnUNet_results is not None else None
        if 'pretrain_name' in plans.keys():
            pretrain_name = plans['pretrain_name']
        else:
            pretrain_name= None
        if pretrain_name is not None:
            if 'dublicate' in plans.keys():
                dublicate = '_dublicate_' + plans['dublicate']
            else:
                dublicate = ''
            self.output_folder_base = join(nnUNet_results, self.plans_manager.dataset_name, 'pretrained', pretrain_name + dublicate,
                                           self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" + configuration) \
                if nnUNet_results is not None else None
            self.output_folder = join(self.output_folder_base, f'fold_{fold}')

        else:
            raise RuntimeError('No pretraining weights provided. Makes no sense to use this trainier')

        self.preprocessed_dataset_folder = join(self.preprocessed_dataset_folder_base,
                                                self.configuration_manager.data_identifier)
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
        self.initial_lr = 1e-2
        self.weight_decay = 3e-5
        self.oversample_foreground_percent = 0.33
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self.num_epochs = 1000
        self.current_epoch = 0
        self.enable_deep_supervision = True


        ### Dealing with labels/regions
        self.label_manager = self.plans_manager.get_label_manager(dataset_json)
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
        self.logger = nnUNetLogger()

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
class nnUNetTrainer_warmupsegheads(nnUNetTrainer_pretrained):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        #### hyperparameter for warmup
        self.warmup_duration_whole_net = 50  # lin increase whole network
        self.warmup_heads_max_lr = 1e-3  # lin increase lr for heads
        self.warmup_duration_heads = 10  # this is for the seg heads
        self.num_epochs = 1000 + self.warmup_duration_whole_net + self.warmup_duration_heads

    def initialize(self):
        if not self.was_initialized:
            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)

            self.network = self.build_network_architecture(self.plans_manager, self.dataset_json,
                                                                        self.configuration_manager,
                                                                        self.num_input_channels,
                                                                        enable_deep_supervision=True).to(self.device)
            # compile network for free speedup
            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers(True)
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss = self._build_loss()
            # torch 2.2.2 crashes upon compiling CE loss
            # if self._do_i_compile():
            #     self.loss = torch.compile(self.loss)
            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")

    def configure_optimizers(self, seg_heads_only=False, netwarmup=False):

        if isinstance(self.network, DDP):
            params = self.network.module.parameters()
            heads = self.network.module.decoder.seg_layers.parameters()
        else:
            params = self.network.parameters()
            # print(self.network.state_dict().keys())
            heads = self.network.decoder.seg_layers.parameters()
        if seg_heads_only:
            self.print_to_log_file("train only heads")
            optimizer = torch.optim.SGD(list(heads), self.initial_lr, weight_decay=self.weight_decay,
                                        momentum=0.99, nesterov=True)
            lr_scheduler = Lin_incr_LRScheduler(optimizer, self.warmup_heads_max_lr, self.warmup_duration_heads)
        elif netwarmup and not seg_heads_only:
            self.print_to_log_file("train whole net, warmup")
            optimizer = torch.optim.SGD(params, self.initial_lr, weight_decay=self.weight_decay,
                                        momentum=0.99, nesterov=True)
            lr_scheduler = Lin_incr_offset_LRScheduler(optimizer, self.initial_lr, self.warmup_duration_whole_net, self.warmup_duration_heads)
        else:
            self.print_to_log_file("train whole net, default schedule")
            optimizer = torch.optim.SGD(params, self.initial_lr, weight_decay=self.weight_decay,
                                        momentum=0.99, nesterov=True)
            lr_scheduler = PolyLRScheduler_offset(optimizer, self.initial_lr, self.num_epochs, self.warmup_duration_whole_net + self.warmup_duration_heads)
        return optimizer, lr_scheduler

    def on_train_epoch_start(self):
        self.network.train()
        if self.current_epoch == self.warmup_duration_heads:
            self.optimizer, self.lr_scheduler = self.configure_optimizers(False, True)
        if self.current_epoch == self.warmup_duration_heads + self.warmup_duration_whole_net:
            self.optimizer, self.lr_scheduler = self.configure_optimizers(False, False)
        self.lr_scheduler.step(self.current_epoch)
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch}')
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}")
        # lrs are the same for all workers so we don't need to gather them in case of DDP training
        self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)

    def on_epoch_start(self):
        # if isinstance(self.network, DDP):
        #     self.print_to_log_file('seg output weight', self.network.module.decoder.seg_layers[0].weight[0,0])
        #     if self.has_stem:
        #         self.print_to_log_file('stem weight', self.network.module.encoder.stem.convs[0].conv.weight[0, 0])
        #         self.print_to_log_file('first fixed weight', self.network.module.encoder.stages[0].blocks[0].conv1.conv.weight[0, 0, 0])
        #     else:
        #         self.print_to_log_file('stem weight',self.network.module.encoder.stages[0][0].convs[0].conv.weight[0, 0, 0])
        #         self.print_to_log_file('first fixed weight', self.network.module.encoder.stages[0][0].convs[1].conv.weight[0, 0, 0])
        # else:
        #     self.print_to_log_file('seg output weight', self.network.decoder.seg_layers[0].weight[0,0])
        #     if self.has_stem:
        #         self.print_to_log_file('stem weight', self.network.encoder.stem.convs[0].conv.weight[0, 0])
        #         self.print_to_log_file('free weight', self.network.encoder.stages[0].blocks[0].conv1.conv.weight[0, 0, 0])
        #     else:
        #         self.print_to_log_file('stem weight',self.network.encoder.stages[0][0].convs[0].conv.weight[0, 0, 0])
        #         self.print_to_log_file('first fixed', self.network.encoder.stages[0][0].convs[1].conv.weight[0, 0, 0])


        self.logger.log('epoch_start_timestamps', time(), self.current_epoch)

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

        if self.warmup_duration_heads + self.warmup_duration_whole_net >= self.current_epoch >= self.warmup_duration_heads:
            self.optimizer, self.lr_scheduler = self.configure_optimizers(False, True)
        if self.current_epoch >= self.warmup_duration_heads + self.warmup_duration_whole_net:
            self.optimizer, self.lr_scheduler = self.configure_optimizers(False, False)

        self.optimizer.load_state_dict(checkpoint['optimizer_state'])



        if self.grad_scaler is not None:
            if checkpoint['grad_scaler_state'] is not None:
                self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])
class nnUNetTrainer_warmupsegheads_20img(nnUNetTrainer_warmupsegheads):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        self.num_train_cases = 20
    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.json file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.json file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            case_identifiers = get_case_identifiers(self.preprocessed_dataset_folder)
            tr_keys = case_identifiers
            val_keys = tr_keys
        else:
            splits_file = join(self.preprocessed_dataset_folder_base, "splits_final.json")
            print(self.preprocessed_dataset_folder)
            dataset = nnUNetDataset(self.preprocessed_dataset_folder, case_identifiers=None,
                                    num_images_properties_loading_threshold=0,
                                    folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                all_keys_sorted = list(np.sort(list(dataset.keys())))

                splits = generate_crossval_split(all_keys_sorted, seed=12345, n_splits=5)
                save_json(splits, splits_file)

            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_json(splits_file)
                self.print_to_log_file(f"The split file contains {len(splits)} splits.")

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


        rnd = np.random.RandomState(seed=12345 + self.fold)
        idx_subsample = rnd.choice(len(tr_keys),  self.num_train_cases, replace=False)
        tr_keys = [tr_keys[i] for i in idx_subsample]
        self.print_to_log_file('We train only with the following %d cases' % self.num_train_cases)

        return tr_keys, val_keys
class nnUNetTrainer_warmupsegheads_nosmooth(nnUNetTrainer_warmupsegheads):
    def _build_loss(self):
        # set smooth to 0
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss({},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 0, 'ddp': self.is_ddp},
                                   use_ignore_label=self.label_manager.ignore_label is not None,
                                   dice_class=MemoryEfficientSoftDiceLoss)
        else:
            loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 0, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                                  ignore_label=self.label_manager.ignore_label,
                                  dice_class=MemoryEfficientSoftDiceLoss)

        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)
        return loss
class nnUNetTrainer_warmupsegheads_nosmooth_20img(nnUNetTrainer_warmupsegheads_nosmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        self.num_train_cases = 20
    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.json file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.json file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            case_identifiers = get_case_identifiers(self.preprocessed_dataset_folder)
            tr_keys = case_identifiers
            val_keys = tr_keys
        else:
            splits_file = join(self.preprocessed_dataset_folder_base, "splits_final.json")
            print(self.preprocessed_dataset_folder)
            dataset = nnUNetDataset(self.preprocessed_dataset_folder, case_identifiers=None,
                                    num_images_properties_loading_threshold=0,
                                    folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                all_keys_sorted = list(np.sort(list(dataset.keys())))

                splits = generate_crossval_split(all_keys_sorted, seed=12345, n_splits=5)
                save_json(splits, splits_file)

            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_json(splits_file)
                self.print_to_log_file(f"The split file contains {len(splits)} splits.")

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


        rnd = np.random.RandomState(seed=12345 + self.fold)
        idx_subsample = rnd.choice(len(tr_keys),  self.num_train_cases, replace=False)
        tr_keys = [tr_keys[i] for i in idx_subsample]
        self.print_to_log_file('We train only with the following %d cases' % self.num_train_cases)

        return tr_keys, val_keys
class nnUNetTrainer_warmupsegheads_lr1e3(nnUNetTrainer_warmupsegheads):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-3
class nnUNetTrainer_warmupsegheads_lr1e3_20img(nnUNetTrainer_warmupsegheads_lr1e3):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        self.num_train_cases = 20
    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.json file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.json file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            case_identifiers = get_case_identifiers(self.preprocessed_dataset_folder)
            tr_keys = case_identifiers
            val_keys = tr_keys
        else:
            splits_file = join(self.preprocessed_dataset_folder_base, "splits_final.json")
            print(self.preprocessed_dataset_folder)
            dataset = nnUNetDataset(self.preprocessed_dataset_folder, case_identifiers=None,
                                    num_images_properties_loading_threshold=0,
                                    folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                all_keys_sorted = list(np.sort(list(dataset.keys())))

                splits = generate_crossval_split(all_keys_sorted, seed=12345, n_splits=5)
                save_json(splits, splits_file)

            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_json(splits_file)
                self.print_to_log_file(f"The split file contains {len(splits)} splits.")

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


        rnd = np.random.RandomState(seed=12345 + self.fold)
        idx_subsample = rnd.choice(len(tr_keys),  self.num_train_cases, replace=False)
        tr_keys = [tr_keys[i] for i in idx_subsample]
        self.print_to_log_file('We train only with the following %d cases' % self.num_train_cases)

        return tr_keys, val_keys
class nnUNetTrainer_warmupsegheads_lr1e3_nosmooth(nnUNetTrainer_warmupsegheads_nosmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-3
class nnUNetTrainer_warmupsegheads_nosmooth_20img(nnUNetTrainer_warmupsegheads_lr1e3_nosmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        self.num_train_cases = 20
    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.json file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.json file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            case_identifiers = get_case_identifiers(self.preprocessed_dataset_folder)
            tr_keys = case_identifiers
            val_keys = tr_keys
        else:
            splits_file = join(self.preprocessed_dataset_folder_base, "splits_final.json")
            print(self.preprocessed_dataset_folder)
            dataset = nnUNetDataset(self.preprocessed_dataset_folder, case_identifiers=None,
                                    num_images_properties_loading_threshold=0,
                                    folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                all_keys_sorted = list(np.sort(list(dataset.keys())))

                splits = generate_crossval_split(all_keys_sorted, seed=12345, n_splits=5)
                save_json(splits, splits_file)

            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_json(splits_file)
                self.print_to_log_file(f"The split file contains {len(splits)} splits.")

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


        rnd = np.random.RandomState(seed=12345 + self.fold)
        idx_subsample = rnd.choice(len(tr_keys),  self.num_train_cases, replace=False)
        tr_keys = [tr_keys[i] for i in idx_subsample]
        self.print_to_log_file('We train only with the following %d cases' % self.num_train_cases)

        return tr_keys, val_keys
class nnUNetTrainer_lr1e3(nnUNetTrainer_pretrained):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-3

class nnUNetTrainer_lr1e3_20img(nnUNetTrainer_lr1e3):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        self.num_train_cases = 20
    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.json file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.json file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            case_identifiers = get_case_identifiers(self.preprocessed_dataset_folder)
            tr_keys = case_identifiers
            val_keys = tr_keys
        else:
            splits_file = join(self.preprocessed_dataset_folder_base, "splits_final.json")
            print(self.preprocessed_dataset_folder)
            dataset = nnUNetDataset(self.preprocessed_dataset_folder, case_identifiers=None,
                                    num_images_properties_loading_threshold=0,
                                    folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                all_keys_sorted = list(np.sort(list(dataset.keys())))

                splits = generate_crossval_split(all_keys_sorted, seed=12345, n_splits=5)
                save_json(splits, splits_file)

            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_json(splits_file)
                self.print_to_log_file(f"The split file contains {len(splits)} splits.")

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


        rnd = np.random.RandomState(seed=12345 + self.fold)
        idx_subsample = rnd.choice(len(tr_keys),  self.num_train_cases, replace=False)
        tr_keys = [tr_keys[i] for i in idx_subsample]
        self.print_to_log_file('We train only with the following %d cases' % self.num_train_cases)

        return tr_keys, val_keys

class nnUNetTrainer_lr1e3_nosmooth(nnUNetTrainer_pretrained_nosmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-3
class nnUNetTrainer_pretrained_encoder(nnUNetTrainer_pretrained):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
class nnUNetTrainer_pretrained_encoder_20img(nnUNetTrainer_pretrained_encoder):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        self.num_train_cases = 20

    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.json file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.json file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            case_identifiers = get_case_identifiers(self.preprocessed_dataset_folder)
            tr_keys = case_identifiers
            val_keys = tr_keys
        else:
            splits_file = join(self.preprocessed_dataset_folder_base, "splits_final.json")
            print(self.preprocessed_dataset_folder)
            dataset = nnUNetDataset(self.preprocessed_dataset_folder, case_identifiers=None,
                                    num_images_properties_loading_threshold=0,
                                    folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                all_keys_sorted = list(np.sort(list(dataset.keys())))

                splits = generate_crossval_split(all_keys_sorted, seed=12345, n_splits=5)
                save_json(splits, splits_file)

            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_json(splits_file)
                self.print_to_log_file(f"The split file contains {len(splits)} splits.")

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

        rnd = np.random.RandomState(seed=12345 + self.fold)
        idx_subsample = rnd.choice(len(tr_keys), self.num_train_cases, replace=False)
        tr_keys = [tr_keys[i] for i in idx_subsample]
        self.print_to_log_file('We train only with the following %d cases' % self.num_train_cases)

        return tr_keys, val_keys


class nnUNetTrainer_pretrained_encoder_lr1e3(nnUNetTrainer_lr1e3):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

class nnUNetTrainer_pretrained_encoder_lr1e3_20img(nnUNetTrainer_pretrained_encoder_lr1e3):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        self.num_train_cases = 20

    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.json file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.json file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            case_identifiers = get_case_identifiers(self.preprocessed_dataset_folder)
            tr_keys = case_identifiers
            val_keys = tr_keys
        else:
            splits_file = join(self.preprocessed_dataset_folder_base, "splits_final.json")
            print(self.preprocessed_dataset_folder)
            dataset = nnUNetDataset(self.preprocessed_dataset_folder, case_identifiers=None,
                                    num_images_properties_loading_threshold=0,
                                    folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                all_keys_sorted = list(np.sort(list(dataset.keys())))

                splits = generate_crossval_split(all_keys_sorted, seed=12345, n_splits=5)
                save_json(splits, splits_file)

            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_json(splits_file)
                self.print_to_log_file(f"The split file contains {len(splits)} splits.")

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

        rnd = np.random.RandomState(seed=12345 + self.fold)
        idx_subsample = rnd.choice(len(tr_keys), self.num_train_cases, replace=False)
        tr_keys = [tr_keys[i] for i in idx_subsample]
        self.print_to_log_file('We train only with the following %d cases' % self.num_train_cases)

        return tr_keys, val_keys
class nnUNetTrainer_warmupdecoder(nnUNetTrainer_pretrained):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        #### hyperparameter for warmup
        self.warmup_duration_whole_net = 50  # lin increase whole network
        self.warmup_heads_max_lr = 1e-3  # lin increase lr for heads
        self.warmup_duration_decoder= 50  # this is for the DECODER
        self.num_epochs = 1000 + self.warmup_duration_whole_net + self.warmup_duration_decoder



    def initialize(self):
        if not self.was_initialized:
            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)

            self.network = self.build_network_architecture(
                self.configuration_manager.network_arch_class_name,
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                self.label_manager.num_segmentation_heads,
                self.enable_deep_supervision
            ).to(self.device)
            # compile network for free speedup
            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers(True)
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss = self._build_loss()
            # torch 2.2.2 crashes upon compiling CE loss
            # if self._do_i_compile():
            #     self.loss = torch.compile(self.loss)
            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")

    def configure_optimizers(self, dec_only=False, netwarmup=False):

        if isinstance(self.network, DDP):
            params = self.network.module.parameters()
            dec = self.network.module.decoder.parameters()
        else:
            params = self.network.parameters()
            # print(self.network.state_dict().keys())
            dec = self.network.decoder.parameters()
        if dec_only:
            self.print_to_log_file("train only decoder")
            optimizer = torch.optim.SGD(list(dec), self.initial_lr, weight_decay=self.weight_decay,
                                        momentum=0.99, nesterov=True)
            lr_scheduler = Lin_incr_LRScheduler(optimizer, self.warmup_heads_max_lr, self.warmup_duration_decoder)
        elif netwarmup and not dec_only:
            self.print_to_log_file("train whole net, warmup")
            optimizer = torch.optim.SGD(params, self.initial_lr, weight_decay=self.weight_decay,
                                        momentum=0.99, nesterov=True)
            lr_scheduler = Lin_incr_offset_LRScheduler(optimizer, self.initial_lr, self.warmup_duration_whole_net, self.warmup_duration_decoder)
        else:
            self.print_to_log_file("train whole net, default schedule")
            optimizer = torch.optim.SGD(params, self.initial_lr, weight_decay=self.weight_decay,
                                        momentum=0.99, nesterov=True)
            lr_scheduler = PolyLRScheduler_offset(optimizer, self.initial_lr, self.num_epochs, self.warmup_duration_whole_net + self.warmup_duration_decoder)
        return optimizer, lr_scheduler

    def on_train_epoch_start(self):
        self.network.train()
        if self.current_epoch == self.warmup_duration_decoder:
            self.optimizer, self.lr_scheduler = self.configure_optimizers(False, True)
        if self.current_epoch == self.warmup_duration_decoder + self.warmup_duration_whole_net:
            self.optimizer, self.lr_scheduler = self.configure_optimizers(False, False)
        self.lr_scheduler.step(self.current_epoch)
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch}')
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}")
        # lrs are the same for all workers so we don't need to gather them in case of DDP training
        self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)

    def on_epoch_start(self):
        # if isinstance(self.network, DDP):
        #     self.print_to_log_file('seg output weight', self.network.module.decoder.seg_layers[0].weight[0,0])
        #     if self.has_stem:
        #         self.print_to_log_file('stem weight', self.network.module.encoder.stem.convs[0].conv.weight[0, 0])
        #         self.print_to_log_file('first fixed weight', self.network.module.encoder.stages[0].blocks[0].conv1.conv.weight[0, 0, 0])
        #     else:
        #         self.print_to_log_file('stem weight',self.network.module.encoder.stages[0][0].convs[0].conv.weight[0, 0, 0])
        #         self.print_to_log_file('first fixed weight', self.network.module.encoder.stages[0][0].convs[1].conv.weight[0, 0, 0])
        # else:
        #     self.print_to_log_file('seg output weight', self.network.decoder.seg_layers[0].weight[0,0])
        #     if self.has_stem:
        #         self.print_to_log_file('stem weight', self.network.encoder.stem.convs[0].conv.weight[0, 0])
        #         self.print_to_log_file('free weight', self.network.encoder.stages[0].blocks[0].conv1.conv.weight[0, 0, 0])
        #     else:
        #         self.print_to_log_file('stem weight',self.network.encoder.stages[0][0].convs[0].conv.weight[0, 0, 0])
        #         self.print_to_log_file('first fixed', self.network.encoder.stages[0][0].convs[1].conv.weight[0, 0, 0])


        self.logger.log('epoch_start_timestamps', time(), self.current_epoch)

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

        if self.warmup_duration_decoder + self.warmup_duration_whole_net >= self.current_epoch >= self.warmup_duration_decoder:
            self.optimizer, self.lr_scheduler = self.configure_optimizers(False, True)
        if self.current_epoch >= self.warmup_duration_decoder + self.warmup_duration_whole_net:
            self.optimizer, self.lr_scheduler = self.configure_optimizers(False, False)

        self.optimizer.load_state_dict(checkpoint['optimizer_state'])



        if self.grad_scaler is not None:
            if checkpoint['grad_scaler_state'] is not None:
                self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])

class nnUNetTrainer_warmupdecoder_20img(nnUNetTrainer_warmupdecoder):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        self.num_train_cases = 20

    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.json file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.json file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            case_identifiers = get_case_identifiers(self.preprocessed_dataset_folder)
            tr_keys = case_identifiers
            val_keys = tr_keys
        else:
            splits_file = join(self.preprocessed_dataset_folder_base, "splits_final.json")
            print(self.preprocessed_dataset_folder)
            dataset = nnUNetDataset(self.preprocessed_dataset_folder, case_identifiers=None,
                                    num_images_properties_loading_threshold=0,
                                    folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                all_keys_sorted = list(np.sort(list(dataset.keys())))

                splits = generate_crossval_split(all_keys_sorted, seed=12345, n_splits=5)
                save_json(splits, splits_file)

            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_json(splits_file)
                self.print_to_log_file(f"The split file contains {len(splits)} splits.")

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

        rnd = np.random.RandomState(seed=12345 + self.fold)
        idx_subsample = rnd.choice(len(tr_keys), self.num_train_cases, replace=False)
        tr_keys = [tr_keys[i] for i in idx_subsample]
        self.print_to_log_file('We train only with the following %d cases' % self.num_train_cases)

        return tr_keys, val_keys

class nnUNetTrainer_warmupdecoder_lr1e3(nnUNetTrainer_warmupdecoder):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-3

class nnUNetTrainer_warmupdecoder_lr1e3_20img(nnUNetTrainer_warmupdecoder_lr1e3):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        self.num_train_cases = 20

    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.json file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.json file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            case_identifiers = get_case_identifiers(self.preprocessed_dataset_folder)
            tr_keys = case_identifiers
            val_keys = tr_keys
        else:
            splits_file = join(self.preprocessed_dataset_folder_base, "splits_final.json")
            print(self.preprocessed_dataset_folder)
            dataset = nnUNetDataset(self.preprocessed_dataset_folder, case_identifiers=None,
                                    num_images_properties_loading_threshold=0,
                                    folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                all_keys_sorted = list(np.sort(list(dataset.keys())))

                splits = generate_crossval_split(all_keys_sorted, seed=12345, n_splits=5)
                save_json(splits, splits_file)

            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_json(splits_file)
                self.print_to_log_file(f"The split file contains {len(splits)} splits.")

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

        rnd = np.random.RandomState(seed=12345 + self.fold)
        idx_subsample = rnd.choice(len(tr_keys), self.num_train_cases, replace=False)
        tr_keys = [tr_keys[i] for i in idx_subsample]
        self.print_to_log_file('We train only with the following %d cases' % self.num_train_cases)

        return tr_keys, val_keys
class nnUNetTrainer_warmupdecoder_nosmooth(nnUNetTrainer_warmupdecoder):
    def _build_loss(self):
        # set smooth to 0
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss({},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 0, 'ddp': self.is_ddp},
                                   use_ignore_label=self.label_manager.ignore_label is not None,
                                   dice_class=MemoryEfficientSoftDiceLoss)
        else:
            loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 0, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                                  ignore_label=self.label_manager.ignore_label,
                                  dice_class=MemoryEfficientSoftDiceLoss)

        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)
        return loss

class nnUNetTrainer_warmupfrozendecoder(nnUNetTrainer_pretrained):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        #### hyperparameter for warmup
        self.warmup_duration_decoder = 50  # lin increase whole decoder
        self.warmup_heads_max_lr = 1e-3  # lin increase lr for decoder
        self.num_epochs = 1000 + self.warmup_duration_decoder

    def initialize(self):
        if not self.was_initialized:
            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)

            self.network = self.build_network_architecture(
                self.configuration_manager.network_arch_class_name,
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                self.label_manager.num_segmentation_heads,
                self.enable_deep_supervision
            ).to(self.device)
            # compile network for free speedup
            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers(True, True)
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss = self._build_loss()
            # torch 2.2.2 crashes upon compiling CE loss
            # if self._do_i_compile():
            #     self.loss = torch.compile(self.loss)
            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")

    def configure_optimizers(self, dec_only=False, netwarmup=False):

        if isinstance(self.network, DDP):
            params = self.network.module.parameters()
            dec = self.network.module.decoder.parameters()
        else:
            params = self.network.parameters()
            # print(self.network.state_dict().keys())
            dec = self.network.decoder.parameters()
        if netwarmup and dec_only:
            self.print_to_log_file("train decoder, warmup")
            optimizer = torch.optim.SGD(list(dec), self.initial_lr, weight_decay=self.weight_decay,
                                        momentum=0.99, nesterov=True)
            lr_scheduler = Lin_incr_offset_LRScheduler(optimizer, self.initial_lr, self.warmup_duration_decoder)
        else:
            self.print_to_log_file("train decoder, default schedule")
            optimizer = torch.optim.SGD(list(dec), self.initial_lr, weight_decay=self.weight_decay,
                                        momentum=0.99, nesterov=True)
            lr_scheduler = PolyLRScheduler_offset(optimizer, self.initial_lr, self.num_epochs, self.warmup_duration_decoder)
        return optimizer, lr_scheduler
    def on_train_epoch_start(self):
        self.network.train()
        if self.current_epoch == self.warmup_duration_decoder:
            self.optimizer, self.lr_scheduler = self.configure_optimizers(True, False)
        self.lr_scheduler.step(self.current_epoch)
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch}')
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}")
        # lrs are the same for all workers so we don't need to gather them in case of DDP training
        self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)

    def on_epoch_start(self):
        # if isinstance(self.network, DDP):
        #     self.print_to_log_file('seg output weight', self.network.module.decoder.seg_layers[0].weight[0,0])
        #     if self.has_stem:
        #         self.print_to_log_file('stem weight', self.network.module.encoder.stem.convs[0].conv.weight[0, 0])
        #         self.print_to_log_file('first fixed weight', self.network.module.encoder.stages[0].blocks[0].conv1.conv.weight[0, 0, 0])
        #     else:
        #         self.print_to_log_file('stem weight',self.network.module.encoder.stages[0][0].convs[0].conv.weight[0, 0, 0])
        #         self.print_to_log_file('first fixed weight', self.network.module.encoder.stages[0][0].convs[1].conv.weight[0, 0, 0])
        # else:
        #     self.print_to_log_file('seg output weight', self.network.decoder.seg_layers[0].weight[0,0])
        #     if self.has_stem:
        #         self.print_to_log_file('stem weight', self.network.encoder.stem.convs[0].conv.weight[0, 0])
        #         self.print_to_log_file('free weight', self.network.encoder.stages[0].blocks[0].conv1.conv.weight[0, 0, 0])
        #     else:
        #         self.print_to_log_file('stem weight',self.network.encoder.stages[0][0].convs[0].conv.weight[0, 0, 0])
        #         self.print_to_log_file('first fixed', self.network.encoder.stages[0][0].convs[1].conv.weight[0, 0, 0])


        self.logger.log('epoch_start_timestamps', time(), self.current_epoch)

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

        if self.warmup_duration_decoder >= self.current_epochs:
            self.optimizer, self.lr_scheduler = self.configure_optimizers(True, True)
        if self.current_epoch >= self.warmup_duration_decoder:
            self.optimizer, self.lr_scheduler = self.configure_optimizers(True, False)

        self.optimizer.load_state_dict(checkpoint['optimizer_state'])

        if self.grad_scaler is not None:
            if checkpoint['grad_scaler_state'] is not None:
                self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])

class nnUNetTrainer_warmupfrozendecoder_lr1e3(nnUNetTrainer_warmupfrozendecoder):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        self.initial_lr = 1e-3


class nnUNetTrainer_warmupdecoder_lr1e3_20img(nnUNetTrainer_warmupdecoder_lr1e3):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        self.num_train_cases = 20

    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.json file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.json file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            case_identifiers = get_case_identifiers(self.preprocessed_dataset_folder)
            tr_keys = case_identifiers
            val_keys = tr_keys
        else:
            splits_file = join(self.preprocessed_dataset_folder_base, "splits_final.json")
            print(self.preprocessed_dataset_folder)
            dataset = nnUNetDataset(self.preprocessed_dataset_folder, case_identifiers=None,
                                    num_images_properties_loading_threshold=0,
                                    folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                all_keys_sorted = list(np.sort(list(dataset.keys())))

                splits = generate_crossval_split(all_keys_sorted, seed=12345, n_splits=5)
                save_json(splits, splits_file)

            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_json(splits_file)
                self.print_to_log_file(f"The split file contains {len(splits)} splits.")

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

        rnd = np.random.RandomState(seed=12345 + self.fold)
        idx_subsample = rnd.choice(len(tr_keys), self.num_train_cases, replace=False)
        tr_keys = [tr_keys[i] for i in idx_subsample]
        self.print_to_log_file('We train only with the following %d cases' % self.num_train_cases)

        return tr_keys, val_keys

#
# class nnUNetTrainer_multipleinputs_warmupsegheads_nosmooth(nnUNetTrainerDiceCELoss_noSmooth):
#     def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
#                  device: torch.device = torch.device('cuda')):
#         # From https://grugbrain.dev/. Worth a read ya big brains ;-)
#
#         # apex predator of grug is complexity
#         # complexity bad
#         # say again:
#         # complexity very bad
#         # you say now:
#         # complexity very, very bad
#         # given choice between complexity or one on one against t-rex, grug take t-rex: at least grug see t-rex
#         # complexity is spirit demon that enter codebase through well-meaning but ultimately very clubbable non grug-brain developers and project managers who not fear complexity spirit demon or even know about sometime
#         # one day code base understandable and grug can get work done, everything good!
#         # next day impossible: complexity demon spirit has entered code and very dangerous situation!
#
#         # OK OK I am guilty. But I tried.
#         # https://www.osnews.com/images/comics/wtfm.jpg
#         # https://i.pinimg.com/originals/26/b2/50/26b250a738ea4abc7a5af4d42ad93af0.jpg
#
#         self.is_ddp = dist.is_available() and dist.is_initialized()
#         self.local_rank = 0 if not self.is_ddp else dist.get_rank()
#
#         self.device = device
#
#         # print what device we are using
#         if self.is_ddp:  # implicitly it's clear that we use cuda in this case
#             print(f"I am local rank {self.local_rank}. {device_count()} GPUs are available. The world size is "
#                   f"{dist.get_world_size()}."
#                   f"Setting device to {self.device}")
#             self.device = torch.device(type='cuda', index=self.local_rank)
#         else:
#             if self.device.type == 'cuda':
#                 # we might want to let the user pick this but for now please pick the correct GPU with CUDA_VISIBLE_DEVICES=X
#                 self.device = torch.device(type='cuda', index=0)
#             print(f"Using device: {self.device}")
#
#         # loading and saving this class for continuing from checkpoint should not happen based on pickling. This
#         # would also pickle the network etc. Bad, bad. Instead we just reinstantiate and then load the checkpoint we
#         # need. So let's save the init args
#         self.my_init_kwargs = {}
#         for k in inspect.signature(self.__init__).parameters.keys():
#             self.my_init_kwargs[k] = locals()[k]
#
#         ###  Saving all the init args into class variables for later access
#         self.plans_manager = PlansManager(plans)
#         self.configuration_manager = self.plans_manager.get_configuration(configuration)
#         self.configuration_name = configuration
#         self.dataset_json = dataset_json
#         self.fold = fold
#         self.unpack_dataset = unpack_dataset
#
#         ### Setting all the folder names. We need to make sure things don't crash in case we are just running
#         # inference and some of the folders may not be defined!
#         self.preprocessed_dataset_folder_base = join(nnUNet_preprocessed, self.plans_manager.dataset_name) \
#             if nnUNet_preprocessed is not None else None
#         self.output_folder_base = join(nnUNet_results, self.plans_manager.dataset_name,
#                                        self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" + configuration) \
#             if nnUNet_results is not None else None
#         if 'pretrain_name' in plans.keys():
#             pretrain_name = plans['pretrain_name']
#         else:
#             pretrain_name= None
#         if pretrain_name is not None:
#             self.output_folder_base = join(nnUNet_results, self.plans_manager.dataset_name, 'pretrained', pretrain_name,
#                                            self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" + configuration) \
#                 if nnUNet_results is not None else None
#             self.output_folder = join(self.output_folder_base, f'fold_{fold}')
#
#         else:
#             raise RuntimeError('No pretraining weights provided. Makes no sense to use this trainier')
#
#         self.preprocessed_dataset_folder = join(self.preprocessed_dataset_folder_base,
#                                                 self.configuration_manager.data_identifier)
#         # unlike the previous nnunet folder_with_segs_from_previous_stage is now part of the plans. For now it has to
#         # be a different configuration in the same plans
#         # IMPORTANT! the mapping must be bijective, so lowres must point to fullres and vice versa (using
#         # "previous_stage" and "next_stage"). Otherwise it won't work!
#         self.is_cascaded = self.configuration_manager.previous_stage_name is not None
#         self.folder_with_segs_from_previous_stage = \
#             join(nnUNet_results, self.plans_manager.dataset_name,
#                  self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" +
#                  self.configuration_manager.previous_stage_name, 'predicted_next_stage', self.configuration_name) \
#                 if self.is_cascaded else None
#
#         ### Some hyperparameters for you to fiddle with
#         self.initial_lr = 1e-2
#         self.weight_decay = 3e-5
#         self.oversample_foreground_percent = 0.33
#         self.num_iterations_per_epoch = 250
#         self.num_val_iterations_per_epoch = 50
#         self.num_epochs = 1000
#         self.current_epoch = 0
#         self.enable_deep_supervision = True
#
#         #### hyperparameter for warmup
#         self.warmup_duration_whole_net = 50  # lin increase whole network
#         self.warmup_heads_max_lr = 1e-3  # lin increase lr for heads
#         self.warmup_duration_heads = 10  # this is for the seg heads
#         self.num_epochs = 1000 + self.warmup_duration_whole_net + self.warmup_duration_heads
#         self.has_stem = False if self.configuration_manager.network_arch_class_name=="PlainConvUNet" else True
#         ### Dealing with labels/regions
#         self.label_manager = self.plans_manager.get_label_manager(dataset_json)
#         # labels can either be a list of int (regular training) or a list of tuples of int (region-based training)
#         # needed for predictions. We do sigmoid in case of (overlapping) regions
#
#         self.num_input_channels = None  # -> self.initialize()
#         self.network = None  # -> self.build_network_architecture()
#         self.optimizer = self.lr_scheduler = None  # -> self.initialize
#         self.grad_scaler = GradScaler() if self.device.type == 'cuda' else None
#         self.loss = None  # -> self.initialize
#
#         ### Simple logging. Don't take that away from me!
#         # initialize log file. This is just our log for the print statements etc. Not to be confused with lightning
#         # logging
#         timestamp = datetime.now()
#         maybe_mkdir_p(self.output_folder)
#         self.log_file = join(self.output_folder, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
#                              (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
#                               timestamp.second))
#         self.logger = nnUNetLogger()
#
#         ### placeholders
#         self.dataloader_train = self.dataloader_val = None  # see on_train_start
#
#         ### initializing stuff for remembering things and such
#         self._best_ema = None
#
#         ### inference things
#         self.inference_allowed_mirroring_axes = None  # this variable is set in
#         # self.configure_rotation_dummyDA_mirroring_and_inital_patch_size and will be saved in checkpoints
#
#         ### checkpoint saving stuff
#         self.save_every = 50
#         self.disable_checkpointing = False
#
#         ## DDP batch size and oversampling can differ between workers and needs adaptation
#         # we need to change the batch size in DDP because we don't use any of those distributed samplers
#         self._set_batch_size_and_oversample()
#
#         self.was_initialized = False
#
#         self.print_to_log_file("\n#######################################################################\n"
#                                "Please cite the following paper when using nnU-Net:\n"
#                                "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
#                                "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
#                                "Nature methods, 18(2), 203-211.\n"
#                                "#######################################################################\n",
#                                also_print_to_console=True, add_timestamp=False)
#
#
#     def initialize(self):
#         if not self.was_initialized:
#             self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
#                                                                    self.dataset_json)
#
#             self.network = self.build_network_architecture(
#                 self.configuration_manager.network_arch_class_name,
#                 self.configuration_manager.network_arch_init_kwargs,
#                 self.configuration_manager.network_arch_init_kwargs_req_import,
#                 self.num_input_channels,
#                 self.label_manager.num_segmentation_heads,
#                 self.enable_deep_supervision
#             ).to(self.device)
#             # compile network for free speedup
#             if self._do_i_compile():
#                 self.print_to_log_file('Using torch.compile...')
#                 self.network = torch.compile(self.network)
#
#             self.optimizer, self.lr_scheduler = self.configure_optimizers(True)
#             # if ddp, wrap in DDP wrapper
#             if self.is_ddp:
#                 self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
#                 self.network = DDP(self.network, device_ids=[self.local_rank])
#
#             self.loss = self._build_loss()
#             # torch 2.2.2 crashes upon compiling CE loss
#             # if self._do_i_compile():
#             #     self.loss = torch.compile(self.loss)
#             self.was_initialized = True
#         else:
#             raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
#                                "That should not happen.")
#
#     def configure_optimizers(self, seg_heads_only=False, netwarmup=False):
#
#         if isinstance(self.network, DDP):
#             params = self.network.module.parameters()
#             heads = self.network.module.decoder.seg_layers.parameters()
#             stem = self.network.module.encoder.stem.parameters()
#         else:
#             params = self.network.parameters()
#             # print(self.network.state_dict().keys())
#             heads = self.network.decoder.seg_layers.parameters()
#             if self.has_stem:
#                 stem = self.network.encoder.stem.parameters()
#             else:
#                 stem = self.network.encoder.stages[0][0].convs[0].parameters()
#         if seg_heads_only:
#             self.print_to_log_file("train only stem and heads")
#             optimizer = torch.optim.SGD(list(stem) +list(heads), self.initial_lr, weight_decay=self.weight_decay,
#                                         momentum=0.99, nesterov=True)
#             lr_scheduler = Lin_incr_LRScheduler(optimizer, self.warmup_heads_max_lr, self.warmup_duration_heads)
#         elif netwarmup and not seg_heads_only:
#             self.print_to_log_file("train whole net, warmup")
#             optimizer = torch.optim.SGD(params, self.initial_lr, weight_decay=self.weight_decay,
#                                         momentum=0.99, nesterov=True)
#             lr_scheduler = Lin_incr_offset_LRScheduler(optimizer, self.initial_lr, self.warmup_duration_whole_net, self.warmup_duration_heads)
#         else:
#             self.print_to_log_file("train whole net, default schedule")
#             optimizer = torch.optim.SGD(params, self.initial_lr, weight_decay=self.weight_decay,
#                                         momentum=0.99, nesterov=True)
#             lr_scheduler = PolyLRScheduler_offset(optimizer, self.initial_lr, self.num_epochs, self.warmup_duration_whole_net + self.warmup_duration_heads)
#         return optimizer, lr_scheduler
#
#     def on_train_epoch_start(self):
#         self.network.train()
#         if self.current_epoch == self.warmup_duration_heads:
#             self.optimizer, self.lr_scheduler = self.configure_optimizers(False, True)
#         if self.current_epoch == self.warmup_duration_heads + self.warmup_duration_whole_net:
#             self.optimizer, self.lr_scheduler = self.configure_optimizers(False, False)
#         self.lr_scheduler.step(self.current_epoch)
#         self.print_to_log_file('')
#         self.print_to_log_file(f'Epoch {self.current_epoch}')
#         self.print_to_log_file(
#             f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}")
#         # lrs are the same for all workers so we don't need to gather them in case of DDP training
#         self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)
#
#     def on_epoch_start(self):
#         # if isinstance(self.network, DDP):
#         #     self.print_to_log_file('seg output weight', self.network.module.decoder.seg_layers[0].weight[0,0])
#         #     if self.has_stem:
#         #         self.print_to_log_file('stem weight', self.network.module.encoder.stem.convs[0].conv.weight[0, 0])
#         #         self.print_to_log_file('first fixed weight', self.network.module.encoder.stages[0].blocks[0].conv1.conv.weight[0, 0, 0])
#         #     else:
#         #         self.print_to_log_file('stem weight',self.network.module.encoder.stages[0][0].convs[0].conv.weight[0, 0, 0])
#         #         self.print_to_log_file('first fixed weight', self.network.module.encoder.stages[0][0].convs[1].conv.weight[0, 0, 0])
#         # else:
#         #     self.print_to_log_file('seg output weight', self.network.decoder.seg_layers[0].weight[0,0])
#         #     if self.has_stem:
#         #         self.print_to_log_file('stem weight', self.network.encoder.stem.convs[0].conv.weight[0, 0])
#         #         self.print_to_log_file('free weight', self.network.encoder.stages[0].blocks[0].conv1.conv.weight[0, 0, 0])
#         #     else:
#         #         self.print_to_log_file('stem weight',self.network.encoder.stages[0][0].convs[0].conv.weight[0, 0, 0])
#         #         self.print_to_log_file('first fixed', self.network.encoder.stages[0][0].convs[1].conv.weight[0, 0, 0])
#
#
#         self.logger.log('epoch_start_timestamps', time(), self.current_epoch)
#
#     def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
#         if not self.was_initialized:
#             self.initialize()
#
#         if isinstance(filename_or_checkpoint, str):
#             checkpoint = torch.load(filename_or_checkpoint, map_location=self.device)
#         # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
#         # match. Use heuristic to make it match
#         new_state_dict = {}
#         for k, value in checkpoint['network_weights'].items():
#             key = k
#             if key not in self.network.state_dict().keys() and key.startswith('module.'):
#                 key = key[7:]
#             new_state_dict[key] = value
#
#         self.my_init_kwargs = checkpoint['init_args']
#         self.current_epoch = checkpoint['current_epoch']
#         self.logger.load_checkpoint(checkpoint['logging'])
#         self._best_ema = checkpoint['_best_ema']
#         self.inference_allowed_mirroring_axes = checkpoint[
#             'inference_allowed_mirroring_axes'] if 'inference_allowed_mirroring_axes' in checkpoint.keys() else self.inference_allowed_mirroring_axes
#
#         # messing with state dict naming schemes. Facepalm.
#         if self.is_ddp:
#             if isinstance(self.network.module, OptimizedModule):
#                 self.network.module._orig_mod.load_state_dict(new_state_dict)
#             else:
#                 self.network.module.load_state_dict(new_state_dict)
#         else:
#             if isinstance(self.network, OptimizedModule):
#                 self.network._orig_mod.load_state_dict(new_state_dict)
#             else:
#                 self.network.load_state_dict(new_state_dict)
#
#         if self.warmup_duration_heads + self.warmup_duration_whole_net >= self.current_epoch >= self.warmup_duration_heads:
#             self.optimizer, self.lr_scheduler = self.configure_optimizers(False, True)
#         if self.current_epoch >= self.warmup_duration_heads + self.warmup_duration_whole_net:
#             self.optimizer, self.lr_scheduler = self.configure_optimizers(False, False)
#
#         self.optimizer.load_state_dict(checkpoint['optimizer_state'])
#
#
#
#         if self.grad_scaler is not None:
#             if checkpoint['grad_scaler_state'] is not None:
#                 self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])
#
#
#     def _build_loss(self):
#         # set smooth to 0
#         if self.label_manager.has_regions:
#             loss = DC_and_BCE_loss({},
#                                    {'batch_dice': self.configuration_manager.batch_dice,
#                                     'do_bg': True, 'smooth': 0, 'ddp': self.is_ddp},
#                                    use_ignore_label=self.label_manager.ignore_label is not None,
#                                    dice_class=MemoryEfficientSoftDiceLoss)
#         else:
#             loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
#                                    'smooth': 0, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
#                                   ignore_label=self.label_manager.ignore_label,
#                                   dice_class=MemoryEfficientSoftDiceLoss)
#
#         deep_supervision_scales = self._get_deep_supervision_scales()
#
#         # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
#         # this gives higher resolution outputs more weight in the loss
#         weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
#
#         # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
#         weights = weights / weights.sum()
#         # now wrap the loss
#         loss = DeepSupervisionWrapper(loss, weights)
#         return loss
#
# class nnUNetTrainer_multipleinputs_warmupdecoder_nosmooth(nnUNetTrainer_warmupdecoder_nosmooth):
#     def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
#                  device: torch.device = torch.device('cuda')):
#         # From https://grugbrain.dev/. Worth a read ya big brains ;-)
#
#         # apex predator of grug is complexity
#         # complexity bad
#         # say again:
#         # complexity very bad
#         # you say now:
#         # complexity very, very bad
#         # given choice between complexity or one on one against t-rex, grug take t-rex: at least grug see t-rex
#         # complexity is spirit demon that enter codebase through well-meaning but ultimately very clubbable non grug-brain developers and project managers who not fear complexity spirit demon or even know about sometime
#         # one day code base understandable and grug can get work done, everything good!
#         # next day impossible: complexity demon spirit has entered code and very dangerous situation!
#
#         # OK OK I am guilty. But I tried.
#         # https://www.osnews.com/images/comics/wtfm.jpg
#         # https://i.pinimg.com/originals/26/b2/50/26b250a738ea4abc7a5af4d42ad93af0.jpg
#
#         self.is_ddp = dist.is_available() and dist.is_initialized()
#         self.local_rank = 0 if not self.is_ddp else dist.get_rank()
#
#         self.device = device
#
#         # print what device we are using
#         if self.is_ddp:  # implicitly it's clear that we use cuda in this case
#             print(f"I am local rank {self.local_rank}. {device_count()} GPUs are available. The world size is "
#                   f"{dist.get_world_size()}."
#                   f"Setting device to {self.device}")
#             self.device = torch.device(type='cuda', index=self.local_rank)
#         else:
#             if self.device.type == 'cuda':
#                 # we might want to let the user pick this but for now please pick the correct GPU with CUDA_VISIBLE_DEVICES=X
#                 self.device = torch.device(type='cuda', index=0)
#             print(f"Using device: {self.device}")
#
#         # loading and saving this class for continuing from checkpoint should not happen based on pickling. This
#         # would also pickle the network etc. Bad, bad. Instead we just reinstantiate and then load the checkpoint we
#         # need. So let's save the init args
#         self.my_init_kwargs = {}
#         for k in inspect.signature(self.__init__).parameters.keys():
#             self.my_init_kwargs[k] = locals()[k]
#
#         ###  Saving all the init args into class variables for later access
#         self.plans_manager = PlansManager(plans)
#         self.configuration_manager = self.plans_manager.get_configuration(configuration)
#         self.configuration_name = configuration
#         self.dataset_json = dataset_json
#         self.fold = fold
#         self.unpack_dataset = unpack_dataset
#
#         ### Setting all the folder names. We need to make sure things don't crash in case we are just running
#         # inference and some of the folders may not be defined!
#         self.preprocessed_dataset_folder_base = join(nnUNet_preprocessed, self.plans_manager.dataset_name) \
#             if nnUNet_preprocessed is not None else None
#         self.output_folder_base = join(nnUNet_results, self.plans_manager.dataset_name,
#                                        self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" + configuration) \
#             if nnUNet_results is not None else None
#         if 'pretrain_name' in plans.keys():
#             pretrain_name = plans['pretrain_name']
#         else:
#             pretrain_name= None
#         if pretrain_name is not None:
#             self.output_folder_base = join(nnUNet_results, self.plans_manager.dataset_name, 'pretrained', pretrain_name,
#                                            self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" + configuration) \
#                 if nnUNet_results is not None else None
#             self.output_folder = join(self.output_folder_base, f'fold_{fold}')
#
#         else:
#             raise RuntimeError('No pretraining weights provided. Makes no sense to use this trainier')
#
#         self.preprocessed_dataset_folder = join(self.preprocessed_dataset_folder_base,
#                                                 self.configuration_manager.data_identifier)
#         # unlike the previous nnunet folder_with_segs_from_previous_stage is now part of the plans. For now it has to
#         # be a different configuration in the same plans
#         # IMPORTANT! the mapping must be bijective, so lowres must point to fullres and vice versa (using
#         # "previous_stage" and "next_stage"). Otherwise it won't work!
#         self.is_cascaded = self.configuration_manager.previous_stage_name is not None
#         self.folder_with_segs_from_previous_stage = \
#             join(nnUNet_results, self.plans_manager.dataset_name,
#                  self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" +
#                  self.configuration_manager.previous_stage_name, 'predicted_next_stage', self.configuration_name) \
#                 if self.is_cascaded else None
#
#         ### Some hyperparameters for you to fiddle with
#         self.initial_lr = 1e-2
#         self.weight_decay = 3e-5
#         self.oversample_foreground_percent = 0.33
#         self.num_iterations_per_epoch = 250
#         self.num_val_iterations_per_epoch = 50
#         self.num_epochs = 1000
#         self.current_epoch = 0
#         self.enable_deep_supervision = True
#
#         #### hyperparameter for warmup
#         self.warmup_duration_whole_net = 50  # lin increase whole network
#         self.warmup_heads_max_lr = 1e-3  # lin increase lr for heads
#         self.warmup_duration_heads = 50  # this is for the DECODER
#         self.num_epochs = 1000 + self.warmup_duration_whole_net + self.warmup_duration_heads
#         self.has_stem = False if self.configuration_manager.network_arch_class_name=="PlainConvUNet" else True
#
#         ### Dealing with labels/regions
#         self.label_manager = self.plans_manager.get_label_manager(dataset_json)
#         # labels can either be a list of int (regular training) or a list of tuples of int (region-based training)
#         # needed for predictions. We do sigmoid in case of (overlapping) regions
#
#         self.num_input_channels = None  # -> self.initialize()
#         self.network = None  # -> self.build_network_architecture()
#         self.optimizer = self.lr_scheduler = None  # -> self.initialize
#         self.grad_scaler = GradScaler() if self.device.type == 'cuda' else None
#         self.loss = None  # -> self.initialize
#
#         ### Simple logging. Don't take that away from me!
#         # initialize log file. This is just our log for the print statements etc. Not to be confused with lightning
#         # logging
#         timestamp = datetime.now()
#         maybe_mkdir_p(self.output_folder)
#         self.log_file = join(self.output_folder, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
#                              (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
#                               timestamp.second))
#         self.logger = nnUNetLogger()
#
#         ### placeholders
#         self.dataloader_train = self.dataloader_val = None  # see on_train_start
#
#         ### initializing stuff for remembering things and such
#         self._best_ema = None
#
#         ### inference things
#         self.inference_allowed_mirroring_axes = None  # this variable is set in
#         # self.configure_rotation_dummyDA_mirroring_and_inital_patch_size and will be saved in checkpoints
#
#         ### checkpoint saving stuff
#         self.save_every = 50
#         self.disable_checkpointing = False
#
#         ## DDP batch size and oversampling can differ between workers and needs adaptation
#         # we need to change the batch size in DDP because we don't use any of those distributed samplers
#         self._set_batch_size_and_oversample()
#
#         self.was_initialized = False
#
#         self.print_to_log_file("\n#######################################################################\n"
#                                "Please cite the following paper when using nnU-Net:\n"
#                                "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
#                                "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
#                                "Nature methods, 18(2), 203-211.\n"
#                                "#######################################################################\n",
#                                also_print_to_console=True, add_timestamp=False)
#
#     def configure_optimizers(self, dec_only=False, netwarmup=False):
#
#         if isinstance(self.network, DDP):
#             params = self.network.module.parameters()
#             dec = self.network.module.decoder.parameters()
#             stem = self.network.module.encoder.stem.parameters()
#         else:
#             params = self.network.parameters()
#             # print(self.network.state_dict().keys())
#             dec = self.network.decoder.parameters()
#             if self.has_stem:
#                 stem = self.network.encoder.stem.parameters()
#             else:
#                 stem = self.network.encoder.stages[0][0].convs[0].parameters()
#         if dec_only:
#             self.print_to_log_file("train only decoder")
#             optimizer = torch.optim.SGD(list(stem) +list(dec), self.initial_lr, weight_decay=self.weight_decay,
#                                         momentum=0.99, nesterov=True)
#             lr_scheduler = Lin_incr_LRScheduler(optimizer, self.warmup_heads_max_lr, self.warmup_duration_heads)
#         elif netwarmup and not dec_only:
#             self.print_to_log_file("train whole net, warmup")
#             optimizer = torch.optim.SGD(params, self.initial_lr, weight_decay=self.weight_decay,
#                                         momentum=0.99, nesterov=True)
#             lr_scheduler = Lin_incr_offset_LRScheduler(optimizer, self.initial_lr, self.warmup_duration_whole_net, self.warmup_duration_heads)
#         else:
#             self.print_to_log_file("train whole net, default schedule")
#             optimizer = torch.optim.SGD(params, self.initial_lr, weight_decay=self.weight_decay,
#                                         momentum=0.99, nesterov=True)
#             lr_scheduler = PolyLRScheduler_offset(optimizer, self.initial_lr, self.num_epochs, self.warmup_duration_whole_net + self.warmup_duration_heads)
#         return optimizer, lr_scheduler
#     def on_epoch_start(self):
#         if isinstance(self.network, DDP):
#             self.print_to_log_file('seg output weight', self.network.module.decoder.seg_layers[0].weight[0,0])
#             if self.has_stem:
#                 self.print_to_log_file('stem weight', self.network.module.encoder.stem.convs[0].conv.weight[0, 0])
#                 self.print_to_log_file('first fixed weight', self.network.module.encoder.stages[0].blocks[0].conv1.conv.weight[0, 0, 0])
#             else:
#                 self.print_to_log_file('stem weight',self.network.module.encoder.stages[0][0].convs[0].conv.weight[0, 0, 0])
#                 self.print_to_log_file('first fixed weight', self.network.module.encoder.stages[0][0].convs[1].conv.weight[0, 0, 0])
#         else:
#             self.print_to_log_file('seg output weight', self.network.decoder.seg_layers[0].weight[0,0])
#             if self.has_stem:
#                 self.print_to_log_file('stem weight', self.network.encoder.stem.convs[0].conv.weight[0, 0])
#                 self.print_to_log_file('first fixed weight', self.network.encoder.stages[0].blocks[0].conv1.conv.weight[0, 0, 0])
#             else:
#                 self.print_to_log_file('stem weight',self.network.encoder.stages[0][0].convs[0].conv.weight[0, 0, 0])
#                 self.print_to_log_file('first fixed weight', self.network.encoder.stages[0][0].convs[1].conv.weight[0, 0, 0])
#
#
#         self.logger.log('epoch_start_timestamps', time(), self.current_epoch)

class nnUNetTrainer_warmupnet(nnUNetTrainer_pretrained):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset =True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        #### hyperparameters for warmup
        self.warmup_duration_whole_net = 50  # lin increase whole network
        self.num_epochs = 1000 + self.warmup_duration_whole_net
        self.training_stage = None  # 'warmup_all', 'train'

    def configure_optimizers(self, stage: str = 'warmup_all'):
        assert stage in ['warmup_all', 'train']

        if self.training_stage == stage:
            return self.optimizer, self.lr_scheduler

        if isinstance(self.network, DDP):
            params = self.network.module.parameters()
        else:
            params = self.network.parameters()

        if stage == 'warmup_all':
            self.print_to_log_file("train whole net, warmup")
            optimizer = torch.optim.SGD(params, self.initial_lr, weight_decay=self.weight_decay,
                                        momentum=0.99, nesterov=True)
            lr_scheduler = Lin_incr_LRScheduler(optimizer, self.initial_lr, self.warmup_duration_whole_net)
            self.print_to_log_file(f"Initialized warmup_all optimizer and lr_scheduler at epoch {self.current_epoch}")
        else:
            self.print_to_log_file("train whole net, default schedule")
            if self.training_stage == 'warmup_all':
                # we can keep the existing optimizer and don't need to create a new one. This will allow us to keep
                # the accumulated momentum terms which already point in a useful driection
                optimizer = self.optimizer
            else:
                optimizer = torch.optim.SGD(params, self.initial_lr, weight_decay=self.weight_decay,
                                            momentum=0.99, nesterov=True)
            lr_scheduler = PolyLRScheduler_offset(optimizer, self.initial_lr, self.num_epochs, self.warmup_duration_whole_net)
            self.print_to_log_file(f"Initialized train optimizer and lr_scheduler at epoch {self.current_epoch}")
        self.training_stage = stage
        empty_cache(self.device)
        return optimizer, lr_scheduler

    def on_train_epoch_start(self):
        if self.current_epoch == 0:
            self.optimizer, self.lr_scheduler = self.configure_optimizers('warmup_all')
        elif self.current_epoch == self.warmup_duration_whole_net:
            self.optimizer, self.lr_scheduler = self.configure_optimizers('train')

        super().on_train_epoch_start()

    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        """
        We need to overwrite that entire function because we need to fiddle the correct optimizer in between
        loading the checkpoint and applying the optimizer states. Yuck.
        """
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
            'inference_allowed_mirroring_axes'] if 'inference_allowed_mirroring_axes' in checkpoint.keys() \
            else self.inference_allowed_mirroring_axes

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

        # it's fine to do this every time we load because configure_optimizers will be a no-op if the correct optimizer
        # and lr scheduler are already set up
        if self.current_epoch < self.warmup_duration_whole_net:
            self.optimizer, self.lr_scheduler = self.configure_optimizers('warmup_all')
        else:
            self.optimizer, self.lr_scheduler = self.configure_optimizers('train')

        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if self.grad_scaler is not None:
            if checkpoint['grad_scaler_state'] is not None:
                self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])

class nnUNetTrainer_warmupnet_20img(nnUNetTrainer_warmupnet):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        self.num_train_cases = 20

    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.json file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.json file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            case_identifiers = get_case_identifiers(self.preprocessed_dataset_folder)
            tr_keys = case_identifiers
            val_keys = tr_keys
        else:
            splits_file = join(self.preprocessed_dataset_folder_base, "splits_final.json")
            print(self.preprocessed_dataset_folder)
            dataset = nnUNetDataset(self.preprocessed_dataset_folder, case_identifiers=None,
                                    num_images_properties_loading_threshold=0,
                                    folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                all_keys_sorted = list(np.sort(list(dataset.keys())))

                splits = generate_crossval_split(all_keys_sorted, seed=12345, n_splits=5)
                save_json(splits, splits_file)

            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_json(splits_file)
                self.print_to_log_file(f"The split file contains {len(splits)} splits.")

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

        rnd = np.random.RandomState(seed=12345 + self.fold)
        idx_subsample = rnd.choice(len(tr_keys), self.num_train_cases, replace=False)
        tr_keys = [tr_keys[i] for i in idx_subsample]
        self.print_to_log_file('We train only with the following %d cases' % self.num_train_cases)

        return tr_keys, val_keys
class nnUNetTrainer_warmupnet_nosmooth(nnUNetTrainer_warmupnet):
    def _build_loss(self):
        # set smooth to 0
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss({},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 0, 'ddp': self.is_ddp},
                                   use_ignore_label=self.label_manager.ignore_label is not None,
                                   dice_class=MemoryEfficientSoftDiceLoss)
        else:
            loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 0, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                                  ignore_label=self.label_manager.ignore_label,
                                  dice_class=MemoryEfficientSoftDiceLoss)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss

class nnUNetTrainer_warmupnet_lr1e3_nosmooth(nnUNetTrainer_warmupnet_nosmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-3
class nnUNetTrainer_warmupnet_1e3_nosmooth(nnUNetTrainer_warmupnet_nosmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-3
class nnUNetTrainer_warmupnet_lr1e3(nnUNetTrainer_warmupnet):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-3
class nnUNetTrainer_warmupnet_1e3(nnUNetTrainer_warmupnet):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-3
class nnUNetTrainer_warmupnet_lr1e3_20img(nnUNetTrainer_warmupnet_lr1e3):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        self.num_train_cases = 20

    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.json file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.json file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            case_identifiers = get_case_identifiers(self.preprocessed_dataset_folder)
            tr_keys = case_identifiers
            val_keys = tr_keys
        else:
            splits_file = join(self.preprocessed_dataset_folder_base, "splits_final.json")
            print(self.preprocessed_dataset_folder)
            dataset = nnUNetDataset(self.preprocessed_dataset_folder, case_identifiers=None,
                                    num_images_properties_loading_threshold=0,
                                    folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                all_keys_sorted = list(np.sort(list(dataset.keys())))

                splits = generate_crossval_split(all_keys_sorted, seed=12345, n_splits=5)
                save_json(splits, splits_file)

            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_json(splits_file)
                self.print_to_log_file(f"The split file contains {len(splits)} splits.")

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

        rnd = np.random.RandomState(seed=12345 + self.fold)
        idx_subsample = rnd.choice(len(tr_keys), self.num_train_cases, replace=False)
        tr_keys = [tr_keys[i] for i in idx_subsample]
        self.print_to_log_file('We train only with the following %d cases' % self.num_train_cases)

        return tr_keys, val_keys
class nnUNetTrainer_warmupnet_lr1e3_nosmooth(nnUNetTrainer_warmupnet_nosmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-3

# import inspect
# from multitalent.utilities.helpers import empty_cache
# from torch import distributed as dist
# from torch.cuda import device_count
# from torch.cuda.amp import GradScaler
# import os
# from multitalent.training.dataloading.utils import get_case_identifiers, unpack_dataset
# from multitalent.utilities.plans_handling.plans_handler import PlansManager
#
# from time import time
# from typing import List, Union, Tuple
# import numpy as np
# import torch
# from torch._dynamo import OptimizedModule
# from multitalent.training.dataloading.utils import get_case_identifiers, unpack_dataset
# from multitalent.utilities.crossval_split import generate_crossval_split
# from multitalent.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
# from multitalent.training.loss.deep_supervision import DeepSupervisionWrapper
# from multitalent.training.loss.dice import  MemoryEfficientSoftDiceLoss
# from multitalent.training.lr_scheduler.polylr import PolyLRScheduler_offset, Lin_incr_LRScheduler, Lin_incr_offset_LRScheduler, PolyLRScheduler
# from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
# from multitalent.utilities.label_handling.label_handling import determine_num_input_channels
# from torch.nn.parallel import DistributedDataParallel as DDP
# from multitalent.training.dataloading.nnunet_dataset import nnUNetDataset
# from multitalent.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
# from multitalent.training.nnUNetTrainer.variants.loss.nnUNetTrainerDiceLoss import nnUNetTrainerDiceCELoss_noSmooth
# from datetime import datetime
# from multitalent.training.logging.nnunet_logger import nnUNetLogger
# from multitalent.paths import nnUNet_preprocessed, nnUNet_results
#
# class nnUNetTrainer_warmupsegheads(nnUNetTrainer):
#     def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
#                  device: torch.device = torch.device('cuda')):
#         # From https://grugbrain.dev/. Worth a read ya big brains ;-)
#
#         # apex predator of grug is complexity
#         # complexity bad
#         # say again:
#         # complexity very bad
#         # you say now:
#         # complexity very, very bad
#         # given choice between complexity or one on one against t-rex, grug take t-rex: at least grug see t-rex
#         # complexity is spirit demon that enter codebase through well-meaning but ultimately very clubbable non grug-brain developers and project managers who not fear complexity spirit demon or even know about sometime
#         # one day code base understandable and grug can get work done, everything good!
#         # next day impossible: complexity demon spirit has entered code and very dangerous situation!
#
#         # OK OK I am guilty. But I tried.
#         # https://www.osnews.com/images/comics/wtfm.jpg
#         # https://i.pinimg.com/originals/26/b2/50/26b250a738ea4abc7a5af4d42ad93af0.jpg
#
#         self.is_ddp = dist.is_available() and dist.is_initialized()
#         self.local_rank = 0 if not self.is_ddp else dist.get_rank()
#
#         self.device = device
#
#         # print what device we are using
#         if self.is_ddp:  # implicitly it's clear that we use cuda in this case
#             print(f"I am local rank {self.local_rank}. {device_count()} GPUs are available. The world size is "
#                   f"{dist.get_world_size()}."
#                   f"Setting device to {self.device}")
#             self.device = torch.device(type='cuda', index=self.local_rank)
#         else:
#             if self.device.type == 'cuda':
#                 # we might want to let the user pick this but for now please pick the correct GPU with CUDA_VISIBLE_DEVICES=X
#                 self.device = torch.device(type='cuda', index=0)
#             print(f"Using device: {self.device}")
#
#         # loading and saving this class for continuing from checkpoint should not happen based on pickling. This
#         # would also pickle the network etc. Bad, bad. Instead we just reinstantiate and then load the checkpoint we
#         # need. So let's save the init args
#         self.my_init_kwargs = {}
#         for k in inspect.signature(self.__init__).parameters.keys():
#             self.my_init_kwargs[k] = locals()[k]
#
#         ###  Saving all the init args into class variables for later access
#         self.plans_manager = PlansManager(plans)
#         self.configuration_manager = self.plans_manager.get_configuration(configuration)
#         self.configuration_name = configuration
#         self.dataset_json = dataset_json
#         self.fold = fold
#         self.unpack_dataset = unpack_dataset
#
#         ### Setting all the folder names. We need to make sure things don't crash in case we are just running
#         # inference and some of the folders may not be defined!
#         self.preprocessed_dataset_folder_base = join(nnUNet_preprocessed, self.plans_manager.dataset_name) \
#             if nnUNet_preprocessed is not None else None
#         self.output_folder_base = join(nnUNet_results, self.plans_manager.dataset_name,
#                                        self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" + configuration) \
#             if nnUNet_results is not None else None
#         if 'pretrain_name' in plans.keys():
#             pretrain_name = plans['pretrain_name']
#         else:
#             pretrain_name= None
#         if pretrain_name is not None:
#             self.output_folder_base = join(nnUNet_results, self.plans_manager.dataset_name, 'pretrained', pretrain_name,
#                                            self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" + configuration) \
#                 if nnUNet_results is not None else None
#             self.output_folder = join(self.output_folder_base, f'fold_{fold}')
#
#         else:
#             raise RuntimeError('No pretraining weights provided. Makes no sense to use this trainier')
#
#         self.preprocessed_dataset_folder = join(self.preprocessed_dataset_folder_base,
#                                                 self.configuration_manager.data_identifier)
#         # unlike the previous nnunet folder_with_segs_from_previous_stage is now part of the plans. For now it has to
#         # be a different configuration in the same plans
#         # IMPORTANT! the mapping must be bijective, so lowres must point to fullres and vice versa (using
#         # "previous_stage" and "next_stage"). Otherwise it won't work!
#         self.is_cascaded = self.configuration_manager.previous_stage_name is not None
#         self.folder_with_segs_from_previous_stage = \
#             join(nnUNet_results, self.plans_manager.dataset_name,
#                  self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" +
#                  self.configuration_manager.previous_stage_name, 'predicted_next_stage', self.configuration_name) \
#                 if self.is_cascaded else None
#
#         ### Some hyperparameters for you to fiddle with
#         self.initial_lr = 1e-2
#         self.weight_decay = 3e-5
#         self.oversample_foreground_percent = 0.33
#         self.num_iterations_per_epoch = 250
#         self.num_val_iterations_per_epoch = 50
#         self.num_epochs = 1000
#         self.current_epoch = 0
#         self.enable_deep_supervision = True
#
#         #### hyperparameter for warmup
#         self.warmup_duration_whole_net = 50  # lin increase whole network
#         self.warmup_heads_max_lr = 1e-3  # lin increase lr for heads
#         self.warmup_duration_heads = 10  # this is for the seg heads
#         self.num_epochs = 1000 + self.warmup_duration_whole_net + self.warmup_duration_heads
#
#         ### Dealing with labels/regions
#         self.label_manager = self.plans_manager.get_label_manager(dataset_json)
#         # labels can either be a list of int (regular training) or a list of tuples of int (region-based training)
#         # needed for predictions. We do sigmoid in case of (overlapping) regions
#
#         self.num_input_channels = None  # -> self.initialize()
#         self.network = None  # -> self.build_network_architecture()
#         self.optimizer = self.lr_scheduler = None  # -> self.initialize
#         self.grad_scaler = GradScaler() if self.device.type == 'cuda' else None
#         self.loss = None  # -> self.initialize
#
#         ### Simple logging. Don't take that away from me!
#         # initialize log file. This is just our log for the print statements etc. Not to be confused with lightning
#         # logging
#         timestamp = datetime.now()
#         maybe_mkdir_p(self.output_folder)
#         self.log_file = join(self.output_folder, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
#                              (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
#                               timestamp.second))
#         self.logger = nnUNetLogger()
#
#         ### placeholders
#         self.dataloader_train = self.dataloader_val = None  # see on_train_start
#
#         ### initializing stuff for remembering things and such
#         self._best_ema = None
#
#         ### inference things
#         self.inference_allowed_mirroring_axes = None  # this variable is set in
#         # self.configure_rotation_dummyDA_mirroring_and_inital_patch_size and will be saved in checkpoints
#
#         ### checkpoint saving stuff
#         self.save_every = 50
#         self.disable_checkpointing = False
#
#         ## DDP batch size and oversampling can differ between workers and needs adaptation
#         # we need to change the batch size in DDP because we don't use any of those distributed samplers
#         self._set_batch_size_and_oversample()
#
#         self.was_initialized = False
#
#         self.print_to_log_file("\n#######################################################################\n"
#                                "Please cite the following paper when using nnU-Net:\n"
#                                "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
#                                "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
#                                "Nature methods, 18(2), 203-211.\n"
#                                "#######################################################################\n",
#                                also_print_to_console=True, add_timestamp=False)
#
#
#     def initialize(self):
#         if not self.was_initialized:
#             self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
#                                                                    self.dataset_json)
#
#             self.network = self.build_network_architecture(
#                 self.configuration_manager.network_arch_class_name,
#                 self.configuration_manager.network_arch_init_kwargs,
#                 self.configuration_manager.network_arch_init_kwargs_req_import,
#                 self.num_input_channels,
#                 self.label_manager.num_segmentation_heads,
#                 self.enable_deep_supervision
#             ).to(self.device)
#             # compile network for free speedup
#             if self._do_i_compile():
#                 self.print_to_log_file('Using torch.compile...')
#                 self.network = torch.compile(self.network)
#
#             self.optimizer, self.lr_scheduler = self.configure_optimizers(True)
#             # if ddp, wrap in DDP wrapper
#             if self.is_ddp:
#                 self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
#                 self.network = DDP(self.network, device_ids=[self.local_rank])
#
#             self.loss = self._build_loss()
#             # torch 2.2.2 crashes upon compiling CE loss
#             # if self._do_i_compile():
#             #     self.loss = torch.compile(self.loss)
#             self.was_initialized = True
#         else:
#             raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
#                                "That should not happen.")
#
#
#
#     def configure_optimizers(self, seg_heads_only=False, netwarmup=False):
#
#         if isinstance(self.network, DDP):
#             params = self.network.module.parameters()
#             heads = self.network.module.decoder.seg_layers.parameters()
#         else:
#             params = self.network.parameters()
#             # print(self.network.state_dict().keys())
#             heads = self.network.decoder.seg_layers.parameters()
#         if seg_heads_only:
#             self.print_to_log_file("train only stem and heads")
#             optimizer = torch.optim.SGD(list(heads), self.initial_lr, weight_decay=self.weight_decay,
#                                         momentum=0.99, nesterov=True)
#             lr_scheduler = Lin_incr_LRScheduler(optimizer, self.warmup_heads_max_lr, self.warmup_duration_heads)
#         elif netwarmup and not seg_heads_only:
#             self.print_to_log_file("train whole net, warmup")
#             optimizer = torch.optim.SGD(params, self.initial_lr, weight_decay=self.weight_decay,
#                                         momentum=0.99, nesterov=True)
#             lr_scheduler = Lin_incr_offset_LRScheduler(optimizer, self.initial_lr, self.warmup_duration_whole_net, self.warmup_duration_heads)
#         else:
#             self.print_to_log_file("train whole net, default schedule")
#             optimizer = torch.optim.SGD(params, self.initial_lr, weight_decay=self.weight_decay,
#                                         momentum=0.99, nesterov=True)
#             lr_scheduler = PolyLRScheduler_offset(optimizer, self.initial_lr, self.num_epochs, self.warmup_duration_whole_net + self.warmup_duration_heads)
#         return optimizer, lr_scheduler
#
#     def on_train_epoch_start(self):
#         self.network.train()
#         if self.current_epoch == self.warmup_duration_heads:
#             self.optimizer, self.lr_scheduler = self.configure_optimizers(False, True)
#         if self.current_epoch == self.warmup_duration_heads + self.warmup_duration_whole_net:
#             self.optimizer, self.lr_scheduler = self.configure_optimizers(False, False)
#         self.lr_scheduler.step(self.current_epoch)
#         self.print_to_log_file('')
#         self.print_to_log_file(f'Epoch {self.current_epoch}')
#         self.print_to_log_file(
#             f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}")
#         # lrs are the same for all workers so we don't need to gather them in case of DDP training
#         self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)
#
#     def on_epoch_start(self):
#         # if isinstance(self.network, DDP):
#         #     self.print_to_log_file('seg output weight', self.network.module.decoder.seg_layers[0].weight[0,0])
#         #     if self.has_stem:
#         #         self.print_to_log_file('stem weight', self.network.module.encoder.stem.convs[0].conv.weight[0, 0])
#         #         self.print_to_log_file('first fixed weight', self.network.module.encoder.stages[0].blocks[0].conv1.conv.weight[0, 0, 0])
#         #     else:
#         #         self.print_to_log_file('stem weight',self.network.module.encoder.stages[0][0].convs[0].conv.weight[0, 0, 0])
#         #         self.print_to_log_file('first fixed weight', self.network.module.encoder.stages[0][0].convs[1].conv.weight[0, 0, 0])
#         # else:
#         #     self.print_to_log_file('seg output weight', self.network.decoder.seg_layers[0].weight[0,0])
#         #     if self.has_stem:
#         #         self.print_to_log_file('stem weight', self.network.encoder.stem.convs[0].conv.weight[0, 0])
#         #         self.print_to_log_file('free weight', self.network.encoder.stages[0].blocks[0].conv1.conv.weight[0, 0, 0])
#         #     else:
#         #         self.print_to_log_file('stem weight',self.network.encoder.stages[0][0].convs[0].conv.weight[0, 0, 0])
#         #         self.print_to_log_file('first fixed', self.network.encoder.stages[0][0].convs[1].conv.weight[0, 0, 0])
#
#
#         self.logger.log('epoch_start_timestamps', time(), self.current_epoch)
#
#     def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
#         if not self.was_initialized:
#             self.initialize()
#
#         if isinstance(filename_or_checkpoint, str):
#             checkpoint = torch.load(filename_or_checkpoint, map_location=self.device)
#         # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
#         # match. Use heuristic to make it match
#         new_state_dict = {}
#         for k, value in checkpoint['network_weights'].items():
#             key = k
#             if key not in self.network.state_dict().keys() and key.startswith('module.'):
#                 key = key[7:]
#             new_state_dict[key] = value
#
#         self.my_init_kwargs = checkpoint['init_args']
#         self.current_epoch = checkpoint['current_epoch']
#         self.logger.load_checkpoint(checkpoint['logging'])
#         self._best_ema = checkpoint['_best_ema']
#         self.inference_allowed_mirroring_axes = checkpoint[
#             'inference_allowed_mirroring_axes'] if 'inference_allowed_mirroring_axes' in checkpoint.keys() else self.inference_allowed_mirroring_axes
#
#         # messing with state dict naming schemes. Facepalm.
#         if self.is_ddp:
#             if isinstance(self.network.module, OptimizedModule):
#                 self.network.module._orig_mod.load_state_dict(new_state_dict)
#             else:
#                 self.network.module.load_state_dict(new_state_dict)
#         else:
#             if isinstance(self.network, OptimizedModule):
#                 self.network._orig_mod.load_state_dict(new_state_dict)
#             else:
#                 self.network.load_state_dict(new_state_dict)
#
#         if self.warmup_duration_heads + self.warmup_duration_whole_net >= self.current_epoch >= self.warmup_duration_heads:
#             self.optimizer, self.lr_scheduler = self.configure_optimizers(False, True)
#         if self.current_epoch >= self.warmup_duration_heads + self.warmup_duration_whole_net:
#             self.optimizer, self.lr_scheduler = self.configure_optimizers(False, False)
#
#         self.optimizer.load_state_dict(checkpoint['optimizer_state'])
#
#
#
#         if self.grad_scaler is not None:
#             if checkpoint['grad_scaler_state'] is not None:
#                 self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])
#
#
#     def _build_loss(self):
#         # set smooth to 0
#         if self.label_manager.has_regions:
#             loss = DC_and_BCE_loss({},
#                                    {'batch_dice': self.configuration_manager.batch_dice,
#                                     'do_bg': True, 'smooth': 0, 'ddp': self.is_ddp},
#                                    use_ignore_label=self.label_manager.ignore_label is not None,
#                                    dice_class=MemoryEfficientSoftDiceLoss)
#         else:
#             loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
#                                    'smooth': 0, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
#                                   ignore_label=self.label_manager.ignore_label,
#                                   dice_class=MemoryEfficientSoftDiceLoss)
#
#         deep_supervision_scales = self._get_deep_supervision_scales()
#
#         # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
#         # this gives higher resolution outputs more weight in the loss
#         weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
#
#         # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
#         weights = weights / weights.sum()
#         # now wrap the loss
#         loss = DeepSupervisionWrapper(loss, weights)
#         return loss
#
# class nnUNetTrainer_warmupsegheads_nosmooth(nnUNetTrainerDiceCELoss_noSmooth):
#     def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
#                  device: torch.device = torch.device('cuda')):
#         # From https://grugbrain.dev/. Worth a read ya big brains ;-)
#
#         # apex predator of grug is complexity
#         # complexity bad
#         # say again:
#         # complexity very bad
#         # you say now:
#         # complexity very, very bad
#         # given choice between complexity or one on one against t-rex, grug take t-rex: at least grug see t-rex
#         # complexity is spirit demon that enter codebase through well-meaning but ultimately very clubbable non grug-brain developers and project managers who not fear complexity spirit demon or even know about sometime
#         # one day code base understandable and grug can get work done, everything good!
#         # next day impossible: complexity demon spirit has entered code and very dangerous situation!
#
#         # OK OK I am guilty. But I tried.
#         # https://www.osnews.com/images/comics/wtfm.jpg
#         # https://i.pinimg.com/originals/26/b2/50/26b250a738ea4abc7a5af4d42ad93af0.jpg
#
#         self.is_ddp = dist.is_available() and dist.is_initialized()
#         self.local_rank = 0 if not self.is_ddp else dist.get_rank()
#
#         self.device = device
#
#         # print what device we are using
#         if self.is_ddp:  # implicitly it's clear that we use cuda in this case
#             print(f"I am local rank {self.local_rank}. {device_count()} GPUs are available. The world size is "
#                   f"{dist.get_world_size()}."
#                   f"Setting device to {self.device}")
#             self.device = torch.device(type='cuda', index=self.local_rank)
#         else:
#             if self.device.type == 'cuda':
#                 # we might want to let the user pick this but for now please pick the correct GPU with CUDA_VISIBLE_DEVICES=X
#                 self.device = torch.device(type='cuda', index=0)
#             print(f"Using device: {self.device}")
#
#         # loading and saving this class for continuing from checkpoint should not happen based on pickling. This
#         # would also pickle the network etc. Bad, bad. Instead we just reinstantiate and then load the checkpoint we
#         # need. So let's save the init args
#         self.my_init_kwargs = {}
#         for k in inspect.signature(self.__init__).parameters.keys():
#             self.my_init_kwargs[k] = locals()[k]
#
#         ###  Saving all the init args into class variables for later access
#         self.plans_manager = PlansManager(plans)
#         self.configuration_manager = self.plans_manager.get_configuration(configuration)
#         self.configuration_name = configuration
#         self.dataset_json = dataset_json
#         self.fold = fold
#         self.unpack_dataset = unpack_dataset
#
#         ### Setting all the folder names. We need to make sure things don't crash in case we are just running
#         # inference and some of the folders may not be defined!
#         self.preprocessed_dataset_folder_base = join(nnUNet_preprocessed, self.plans_manager.dataset_name) \
#             if nnUNet_preprocessed is not None else None
#         self.output_folder_base = join(nnUNet_results, self.plans_manager.dataset_name,
#                                        self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" + configuration) \
#             if nnUNet_results is not None else None
#         if 'pretrain_name' in plans.keys():
#             pretrain_name = plans['pretrain_name']
#         else:
#             pretrain_name= None
#         if pretrain_name is not None:
#             self.output_folder_base = join(nnUNet_results, self.plans_manager.dataset_name, 'pretrained', pretrain_name,
#                                            self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" + configuration) \
#                 if nnUNet_results is not None else None
#             self.output_folder = join(self.output_folder_base, f'fold_{fold}')
#
#         else:
#             raise RuntimeError('No pretraining weights provided. Makes no sense to use this trainier')
#
#         self.preprocessed_dataset_folder = join(self.preprocessed_dataset_folder_base,
#                                                 self.configuration_manager.data_identifier)
#         # unlike the previous nnunet folder_with_segs_from_previous_stage is now part of the plans. For now it has to
#         # be a different configuration in the same plans
#         # IMPORTANT! the mapping must be bijective, so lowres must point to fullres and vice versa (using
#         # "previous_stage" and "next_stage"). Otherwise it won't work!
#         self.is_cascaded = self.configuration_manager.previous_stage_name is not None
#         self.folder_with_segs_from_previous_stage = \
#             join(nnUNet_results, self.plans_manager.dataset_name,
#                  self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" +
#                  self.configuration_manager.previous_stage_name, 'predicted_next_stage', self.configuration_name) \
#                 if self.is_cascaded else None
#
#         ### Some hyperparameters for you to fiddle with
#         self.initial_lr = 1e-2
#         self.weight_decay = 3e-5
#         self.oversample_foreground_percent = 0.33
#         self.num_iterations_per_epoch = 250
#         self.num_val_iterations_per_epoch = 50
#         self.num_epochs = 1000
#         self.current_epoch = 0
#         self.enable_deep_supervision = True
#
#         #### hyperparameter for warmup
#         self.warmup_duration_whole_net = 50  # lin increase whole network
#         self.warmup_heads_max_lr = 1e-3  # lin increase lr for heads
#         self.warmup_duration_heads = 10  # this is for the seg heads
#         self.num_epochs = 1000 + self.warmup_duration_whole_net + self.warmup_duration_heads
#
#         ### Dealing with labels/regions
#         self.label_manager = self.plans_manager.get_label_manager(dataset_json)
#         # labels can either be a list of int (regular training) or a list of tuples of int (region-based training)
#         # needed for predictions. We do sigmoid in case of (overlapping) regions
#
#         self.num_input_channels = None  # -> self.initialize()
#         self.network = None  # -> self.build_network_architecture()
#         self.optimizer = self.lr_scheduler = None  # -> self.initialize
#         self.grad_scaler = GradScaler() if self.device.type == 'cuda' else None
#         self.loss = None  # -> self.initialize
#
#         ### Simple logging. Don't take that away from me!
#         # initialize log file. This is just our log for the print statements etc. Not to be confused with lightning
#         # logging
#         timestamp = datetime.now()
#         maybe_mkdir_p(self.output_folder)
#         self.log_file = join(self.output_folder, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
#                              (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
#                               timestamp.second))
#         self.logger = nnUNetLogger()
#
#         ### placeholders
#         self.dataloader_train = self.dataloader_val = None  # see on_train_start
#
#         ### initializing stuff for remembering things and such
#         self._best_ema = None
#
#         ### inference things
#         self.inference_allowed_mirroring_axes = None  # this variable is set in
#         # self.configure_rotation_dummyDA_mirroring_and_inital_patch_size and will be saved in checkpoints
#
#         ### checkpoint saving stuff
#         self.save_every = 50
#         self.disable_checkpointing = False
#
#         ## DDP batch size and oversampling can differ between workers and needs adaptation
#         # we need to change the batch size in DDP because we don't use any of those distributed samplers
#         self._set_batch_size_and_oversample()
#
#         self.was_initialized = False
#
#         self.print_to_log_file("\n#######################################################################\n"
#                                "Please cite the following paper when using nnU-Net:\n"
#                                "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
#                                "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
#                                "Nature methods, 18(2), 203-211.\n"
#                                "#######################################################################\n",
#                                also_print_to_console=True, add_timestamp=False)
#
#
#     def initialize(self):
#         if not self.was_initialized:
#             self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
#                                                                    self.dataset_json)
#
#             self.network = self.build_network_architecture(
#                 self.configuration_manager.network_arch_class_name,
#                 self.configuration_manager.network_arch_init_kwargs,
#                 self.configuration_manager.network_arch_init_kwargs_req_import,
#                 self.num_input_channels,
#                 self.label_manager.num_segmentation_heads,
#                 self.enable_deep_supervision
#             ).to(self.device)
#             # compile network for free speedup
#             if self._do_i_compile():
#                 self.print_to_log_file('Using torch.compile...')
#                 self.network = torch.compile(self.network)
#
#             self.optimizer, self.lr_scheduler = self.configure_optimizers(True)
#             # if ddp, wrap in DDP wrapper
#             if self.is_ddp:
#                 self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
#                 self.network = DDP(self.network, device_ids=[self.local_rank])
#
#             self.loss = self._build_loss()
#             # torch 2.2.2 crashes upon compiling CE loss
#             # if self._do_i_compile():
#             #     self.loss = torch.compile(self.loss)
#             self.was_initialized = True
#         else:
#             raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
#                                "That should not happen.")
#
#
#
#     def configure_optimizers(self, seg_heads_only=False, netwarmup=False):
#
#         if isinstance(self.network, DDP):
#             params = self.network.module.parameters()
#             heads = self.network.module.decoder.seg_layers.parameters()
#         else:
#             params = self.network.parameters()
#             # print(self.network.state_dict().keys())
#             heads = self.network.decoder.seg_layers.parameters()
#         if seg_heads_only:
#             self.print_to_log_file("train only stem and heads")
#             optimizer = torch.optim.SGD(list(heads), self.initial_lr, weight_decay=self.weight_decay,
#                                         momentum=0.99, nesterov=True)
#             lr_scheduler = Lin_incr_LRScheduler(optimizer, self.warmup_heads_max_lr, self.warmup_duration_heads)
#         elif netwarmup and not seg_heads_only:
#             self.print_to_log_file("train whole net, warmup")
#             optimizer = torch.optim.SGD(params, self.initial_lr, weight_decay=self.weight_decay,
#                                         momentum=0.99, nesterov=True)
#             lr_scheduler = Lin_incr_offset_LRScheduler(optimizer, self.initial_lr, self.warmup_duration_whole_net, self.warmup_duration_heads)
#         else:
#             self.print_to_log_file("train whole net, default schedule")
#             optimizer = torch.optim.SGD(params, self.initial_lr, weight_decay=self.weight_decay,
#                                         momentum=0.99, nesterov=True)
#             lr_scheduler = PolyLRScheduler_offset(optimizer, self.initial_lr, self.num_epochs, self.warmup_duration_whole_net + self.warmup_duration_heads)
#         return optimizer, lr_scheduler
#
#     def on_train_epoch_start(self):
#         self.network.train()
#         if self.current_epoch == self.warmup_duration_heads:
#             self.optimizer, self.lr_scheduler = self.configure_optimizers(False, True)
#         if self.current_epoch == self.warmup_duration_heads + self.warmup_duration_whole_net:
#             self.optimizer, self.lr_scheduler = self.configure_optimizers(False, False)
#         self.lr_scheduler.step(self.current_epoch)
#         self.print_to_log_file('')
#         self.print_to_log_file(f'Epoch {self.current_epoch}')
#         self.print_to_log_file(
#             f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}")
#         # lrs are the same for all workers so we don't need to gather them in case of DDP training
#         self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)
#
#     def on_epoch_start(self):
#         # if isinstance(self.network, DDP):
#         #     self.print_to_log_file('seg output weight', self.network.module.decoder.seg_layers[0].weight[0,0])
#         #     if self.has_stem:
#         #         self.print_to_log_file('stem weight', self.network.module.encoder.stem.convs[0].conv.weight[0, 0])
#         #         self.print_to_log_file('first fixed weight', self.network.module.encoder.stages[0].blocks[0].conv1.conv.weight[0, 0, 0])
#         #     else:
#         #         self.print_to_log_file('stem weight',self.network.module.encoder.stages[0][0].convs[0].conv.weight[0, 0, 0])
#         #         self.print_to_log_file('first fixed weight', self.network.module.encoder.stages[0][0].convs[1].conv.weight[0, 0, 0])
#         # else:
#         #     self.print_to_log_file('seg output weight', self.network.decoder.seg_layers[0].weight[0,0])
#         #     if self.has_stem:
#         #         self.print_to_log_file('stem weight', self.network.encoder.stem.convs[0].conv.weight[0, 0])
#         #         self.print_to_log_file('free weight', self.network.encoder.stages[0].blocks[0].conv1.conv.weight[0, 0, 0])
#         #     else:
#         #         self.print_to_log_file('stem weight',self.network.encoder.stages[0][0].convs[0].conv.weight[0, 0, 0])
#         #         self.print_to_log_file('first fixed', self.network.encoder.stages[0][0].convs[1].conv.weight[0, 0, 0])
#
#
#         self.logger.log('epoch_start_timestamps', time(), self.current_epoch)
#
#     def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
#         if not self.was_initialized:
#             self.initialize()
#
#         if isinstance(filename_or_checkpoint, str):
#             checkpoint = torch.load(filename_or_checkpoint, map_location=self.device)
#         # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
#         # match. Use heuristic to make it match
#         new_state_dict = {}
#         for k, value in checkpoint['network_weights'].items():
#             key = k
#             if key not in self.network.state_dict().keys() and key.startswith('module.'):
#                 key = key[7:]
#             new_state_dict[key] = value
#
#         self.my_init_kwargs = checkpoint['init_args']
#         self.current_epoch = checkpoint['current_epoch']
#         self.logger.load_checkpoint(checkpoint['logging'])
#         self._best_ema = checkpoint['_best_ema']
#         self.inference_allowed_mirroring_axes = checkpoint[
#             'inference_allowed_mirroring_axes'] if 'inference_allowed_mirroring_axes' in checkpoint.keys() else self.inference_allowed_mirroring_axes
#
#         # messing with state dict naming schemes. Facepalm.
#         if self.is_ddp:
#             if isinstance(self.network.module, OptimizedModule):
#                 self.network.module._orig_mod.load_state_dict(new_state_dict)
#             else:
#                 self.network.module.load_state_dict(new_state_dict)
#         else:
#             if isinstance(self.network, OptimizedModule):
#                 self.network._orig_mod.load_state_dict(new_state_dict)
#             else:
#                 self.network.load_state_dict(new_state_dict)
#
#         if self.warmup_duration_heads + self.warmup_duration_whole_net >= self.current_epoch >= self.warmup_duration_heads:
#             self.optimizer, self.lr_scheduler = self.configure_optimizers(False, True)
#         if self.current_epoch >= self.warmup_duration_heads + self.warmup_duration_whole_net:
#             self.optimizer, self.lr_scheduler = self.configure_optimizers(False, False)
#
#         self.optimizer.load_state_dict(checkpoint['optimizer_state'])
#
#
#
#         if self.grad_scaler is not None:
#             if checkpoint['grad_scaler_state'] is not None:
#                 self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])
#
#
#     def _build_loss(self):
#         # set smooth to 0
#         if self.label_manager.has_regions:
#             loss = DC_and_BCE_loss({},
#                                    {'batch_dice': self.configuration_manager.batch_dice,
#                                     'do_bg': True, 'smooth': 0, 'ddp': self.is_ddp},
#                                    use_ignore_label=self.label_manager.ignore_label is not None,
#                                    dice_class=MemoryEfficientSoftDiceLoss)
#         else:
#             loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
#                                    'smooth': 0, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
#                                   ignore_label=self.label_manager.ignore_label,
#                                   dice_class=MemoryEfficientSoftDiceLoss)
#
#         deep_supervision_scales = self._get_deep_supervision_scales()
#
#         # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
#         # this gives higher resolution outputs more weight in the loss
#         weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
#
#         # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
#         weights = weights / weights.sum()
#         # now wrap the loss
#         loss = DeepSupervisionWrapper(loss, weights)
#         return loss
# class nnUNetTrainer_warmupsegheads_lr1e3(nnUNetTrainer_warmupsegheads):
#     def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
#                  device: torch.device = torch.device('cuda')):
#         # From https://grugbrain.dev/. Worth a read ya big brains ;-)
#
#         # apex predator of grug is complexity
#         # complexity bad
#         # say again:
#         # complexity very bad
#         # you say now:
#         # complexity very, very bad
#         # given choice between complexity or one on one against t-rex, grug take t-rex: at least grug see t-rex
#         # complexity is spirit demon that enter codebase through well-meaning but ultimately very clubbable non grug-brain developers and project managers who not fear complexity spirit demon or even know about sometime
#         # one day code base understandable and grug can get work done, everything good!
#         # next day impossible: complexity demon spirit has entered code and very dangerous situation!
#
#         # OK OK I am guilty. But I tried.
#         # https://www.osnews.com/images/comics/wtfm.jpg
#         # https://i.pinimg.com/originals/26/b2/50/26b250a738ea4abc7a5af4d42ad93af0.jpg
#
#         self.is_ddp = dist.is_available() and dist.is_initialized()
#         self.local_rank = 0 if not self.is_ddp else dist.get_rank()
#
#         self.device = device
#
#         # print what device we are using
#         if self.is_ddp:  # implicitly it's clear that we use cuda in this case
#             print(f"I am local rank {self.local_rank}. {device_count()} GPUs are available. The world size is "
#                   f"{dist.get_world_size()}."
#                   f"Setting device to {self.device}")
#             self.device = torch.device(type='cuda', index=self.local_rank)
#         else:
#             if self.device.type == 'cuda':
#                 # we might want to let the user pick this but for now please pick the correct GPU with CUDA_VISIBLE_DEVICES=X
#                 self.device = torch.device(type='cuda', index=0)
#             print(f"Using device: {self.device}")
#
#         # loading and saving this class for continuing from checkpoint should not happen based on pickling. This
#         # would also pickle the network etc. Bad, bad. Instead we just reinstantiate and then load the checkpoint we
#         # need. So let's save the init args
#         self.my_init_kwargs = {}
#         for k in inspect.signature(self.__init__).parameters.keys():
#             self.my_init_kwargs[k] = locals()[k]
#
#         ###  Saving all the init args into class variables for later access
#         self.plans_manager = PlansManager(plans)
#         self.configuration_manager = self.plans_manager.get_configuration(configuration)
#         self.configuration_name = configuration
#         self.dataset_json = dataset_json
#         self.fold = fold
#         self.unpack_dataset = unpack_dataset
#
#         ### Setting all the folder names. We need to make sure things don't crash in case we are just running
#         # inference and some of the folders may not be defined!
#         self.preprocessed_dataset_folder_base = join(nnUNet_preprocessed, self.plans_manager.dataset_name) \
#             if nnUNet_preprocessed is not None else None
#         self.output_folder_base = join(nnUNet_results, self.plans_manager.dataset_name,
#                                        self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" + configuration) \
#             if nnUNet_results is not None else None
#         if 'pretrain_name' in plans.keys():
#             pretrain_name = plans['pretrain_name']
#         else:
#             pretrain_name= None
#         if pretrain_name is not None:
#             self.output_folder_base = join(nnUNet_results, self.plans_manager.dataset_name, 'pretrained', pretrain_name,
#                                            self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" + configuration) \
#                 if nnUNet_results is not None else None
#             self.output_folder = join(self.output_folder_base, f'fold_{fold}')
#
#         else:
#             raise RuntimeError('No pretraining weights provided. Makes no sense to use this trainier')
#
#         self.preprocessed_dataset_folder = join(self.preprocessed_dataset_folder_base,
#                                                 self.configuration_manager.data_identifier)
#         # unlike the previous nnunet folder_with_segs_from_previous_stage is now part of the plans. For now it has to
#         # be a different configuration in the same plans
#         # IMPORTANT! the mapping must be bijective, so lowres must point to fullres and vice versa (using
#         # "previous_stage" and "next_stage"). Otherwise it won't work!
#         self.is_cascaded = self.configuration_manager.previous_stage_name is not None
#         self.folder_with_segs_from_previous_stage = \
#             join(nnUNet_results, self.plans_manager.dataset_name,
#                  self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" +
#                  self.configuration_manager.previous_stage_name, 'predicted_next_stage', self.configuration_name) \
#                 if self.is_cascaded else None
#
#         ### Some hyperparameters for you to fiddle with
#         self.initial_lr = 1e-3
#         self.weight_decay = 3e-5
#         self.oversample_foreground_percent = 0.33
#         self.num_iterations_per_epoch = 250
#         self.num_val_iterations_per_epoch = 50
#         self.num_epochs = 1000
#         self.current_epoch = 0
#         self.enable_deep_supervision = True
#
#         #### hyperparameter for warmup
#         self.warmup_duration_whole_net = 50  # lin increase whole network
#         self.warmup_heads_max_lr = 1e-3  # lin increase lr for heads
#         self.warmup_duration_heads = 10  # this is for the seg heads
#         self.num_epochs = 1000 + self.warmup_duration_whole_net + self.warmup_duration_heads
#
#         ### Dealing with labels/regions
#         self.label_manager = self.plans_manager.get_label_manager(dataset_json)
#         # labels can either be a list of int (regular training) or a list of tuples of int (region-based training)
#         # needed for predictions. We do sigmoid in case of (overlapping) regions
#
#         self.num_input_channels = None  # -> self.initialize()
#         self.network = None  # -> self.build_network_architecture()
#         self.optimizer = self.lr_scheduler = None  # -> self.initialize
#         self.grad_scaler = GradScaler() if self.device.type == 'cuda' else None
#         self.loss = None  # -> self.initialize
#
#         ### Simple logging. Don't take that away from me!
#         # initialize log file. This is just our log for the print statements etc. Not to be confused with lightning
#         # logging
#         timestamp = datetime.now()
#         maybe_mkdir_p(self.output_folder)
#         self.log_file = join(self.output_folder, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
#                              (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
#                               timestamp.second))
#         self.logger = nnUNetLogger()
#
#         ### placeholders
#         self.dataloader_train = self.dataloader_val = None  # see on_train_start
#
#         ### initializing stuff for remembering things and such
#         self._best_ema = None
#
#         ### inference things
#         self.inference_allowed_mirroring_axes = None  # this variable is set in
#         # self.configure_rotation_dummyDA_mirroring_and_inital_patch_size and will be saved in checkpoints
#
#         ### checkpoint saving stuff
#         self.save_every = 50
#         self.disable_checkpointing = False
#
#         ## DDP batch size and oversampling can differ between workers and needs adaptation
#         # we need to change the batch size in DDP because we don't use any of those distributed samplers
#         self._set_batch_size_and_oversample()
#
#         self.was_initialized = False
#
#         self.print_to_log_file("\n#######################################################################\n"
#                                "Please cite the following paper when using nnU-Net:\n"
#                                "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
#                                "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
#                                "Nature methods, 18(2), 203-211.\n"
#                                "#######################################################################\n",
#                                also_print_to_console=True, add_timestamp=False)
# class nnUNetTrainer_warmupsegheads_lr1e3_nosmooth(nnUNetTrainer_warmupsegheads_nosmooth):
#     def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
#                  device: torch.device = torch.device('cuda')):
#         # From https://grugbrain.dev/. Worth a read ya big brains ;-)
#
#         # apex predator of grug is complexity
#         # complexity bad
#         # say again:
#         # complexity very bad
#         # you say now:
#         # complexity very, very bad
#         # given choice between complexity or one on one against t-rex, grug take t-rex: at least grug see t-rex
#         # complexity is spirit demon that enter codebase through well-meaning but ultimately very clubbable non grug-brain developers and project managers who not fear complexity spirit demon or even know about sometime
#         # one day code base understandable and grug can get work done, everything good!
#         # next day impossible: complexity demon spirit has entered code and very dangerous situation!
#
#         # OK OK I am guilty. But I tried.
#         # https://www.osnews.com/images/comics/wtfm.jpg
#         # https://i.pinimg.com/originals/26/b2/50/26b250a738ea4abc7a5af4d42ad93af0.jpg
#
#         self.is_ddp = dist.is_available() and dist.is_initialized()
#         self.local_rank = 0 if not self.is_ddp else dist.get_rank()
#
#         self.device = device
#
#         # print what device we are using
#         if self.is_ddp:  # implicitly it's clear that we use cuda in this case
#             print(f"I am local rank {self.local_rank}. {device_count()} GPUs are available. The world size is "
#                   f"{dist.get_world_size()}."
#                   f"Setting device to {self.device}")
#             self.device = torch.device(type='cuda', index=self.local_rank)
#         else:
#             if self.device.type == 'cuda':
#                 # we might want to let the user pick this but for now please pick the correct GPU with CUDA_VISIBLE_DEVICES=X
#                 self.device = torch.device(type='cuda', index=0)
#             print(f"Using device: {self.device}")
#
#         # loading and saving this class for continuing from checkpoint should not happen based on pickling. This
#         # would also pickle the network etc. Bad, bad. Instead we just reinstantiate and then load the checkpoint we
#         # need. So let's save the init args
#         self.my_init_kwargs = {}
#         for k in inspect.signature(self.__init__).parameters.keys():
#             self.my_init_kwargs[k] = locals()[k]
#
#         ###  Saving all the init args into class variables for later access
#         self.plans_manager = PlansManager(plans)
#         self.configuration_manager = self.plans_manager.get_configuration(configuration)
#         self.configuration_name = configuration
#         self.dataset_json = dataset_json
#         self.fold = fold
#         self.unpack_dataset = unpack_dataset
#
#         ### Setting all the folder names. We need to make sure things don't crash in case we are just running
#         # inference and some of the folders may not be defined!
#         self.preprocessed_dataset_folder_base = join(nnUNet_preprocessed, self.plans_manager.dataset_name) \
#             if nnUNet_preprocessed is not None else None
#         self.output_folder_base = join(nnUNet_results, self.plans_manager.dataset_name,
#                                        self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" + configuration) \
#             if nnUNet_results is not None else None
#         if 'pretrain_name' in plans.keys():
#             pretrain_name = plans['pretrain_name']
#         else:
#             pretrain_name= None
#         if pretrain_name is not None:
#             self.output_folder_base = join(nnUNet_results, self.plans_manager.dataset_name, 'pretrained', pretrain_name,
#                                            self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" + configuration) \
#                 if nnUNet_results is not None else None
#             self.output_folder = join(self.output_folder_base, f'fold_{fold}')
#
#         else:
#             raise RuntimeError('No pretraining weights provided. Makes no sense to use this trainier')
#
#         self.preprocessed_dataset_folder = join(self.preprocessed_dataset_folder_base,
#                                                 self.configuration_manager.data_identifier)
#         # unlike the previous nnunet folder_with_segs_from_previous_stage is now part of the plans. For now it has to
#         # be a different configuration in the same plans
#         # IMPORTANT! the mapping must be bijective, so lowres must point to fullres and vice versa (using
#         # "previous_stage" and "next_stage"). Otherwise it won't work!
#         self.is_cascaded = self.configuration_manager.previous_stage_name is not None
#         self.folder_with_segs_from_previous_stage = \
#             join(nnUNet_results, self.plans_manager.dataset_name,
#                  self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" +
#                  self.configuration_manager.previous_stage_name, 'predicted_next_stage', self.configuration_name) \
#                 if self.is_cascaded else None
#
#         ### Some hyperparameters for you to fiddle with
#         self.initial_lr = 1e-3
#         self.weight_decay = 3e-5
#         self.oversample_foreground_percent = 0.33
#         self.num_iterations_per_epoch = 250
#         self.num_val_iterations_per_epoch = 50
#         self.num_epochs = 1000
#         self.current_epoch = 0
#         self.enable_deep_supervision = True
#
#         #### hyperparameter for warmup
#         self.warmup_duration_whole_net = 50  # lin increase whole network
#         self.warmup_heads_max_lr = 1e-3  # lin increase lr for heads
#         self.warmup_duration_heads = 10  # this is for the seg heads
#         self.num_epochs = 1000 + self.warmup_duration_whole_net + self.warmup_duration_heads
#
#         ### Dealing with labels/regions
#         self.label_manager = self.plans_manager.get_label_manager(dataset_json)
#         # labels can either be a list of int (regular training) or a list of tuples of int (region-based training)
#         # needed for predictions. We do sigmoid in case of (overlapping) regions
#
#         self.num_input_channels = None  # -> self.initialize()
#         self.network = None  # -> self.build_network_architecture()
#         self.optimizer = self.lr_scheduler = None  # -> self.initialize
#         self.grad_scaler = GradScaler() if self.device.type == 'cuda' else None
#         self.loss = None  # -> self.initialize
#
#         ### Simple logging. Don't take that away from me!
#         # initialize log file. This is just our log for the print statements etc. Not to be confused with lightning
#         # logging
#         timestamp = datetime.now()
#         maybe_mkdir_p(self.output_folder)
#         self.log_file = join(self.output_folder, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
#                              (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
#                               timestamp.second))
#         self.logger = nnUNetLogger()
#
#         ### placeholders
#         self.dataloader_train = self.dataloader_val = None  # see on_train_start
#
#         ### initializing stuff for remembering things and such
#         self._best_ema = None
#
#         ### inference things
#         self.inference_allowed_mirroring_axes = None  # this variable is set in
#         # self.configure_rotation_dummyDA_mirroring_and_inital_patch_size and will be saved in checkpoints
#
#         ### checkpoint saving stuff
#         self.save_every = 50
#         self.disable_checkpointing = False
#
#         ## DDP batch size and oversampling can differ between workers and needs adaptation
#         # we need to change the batch size in DDP because we don't use any of those distributed samplers
#         self._set_batch_size_and_oversample()
#
#         self.was_initialized = False
#
#         self.print_to_log_file("\n#######################################################################\n"
#                                "Please cite the following paper when using nnU-Net:\n"
#                                "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
#                                "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
#                                "Nature methods, 18(2), 203-211.\n"
#                                "#######################################################################\n",
#                                also_print_to_console=True, add_timestamp=False)
# class nnUNetTrainer_lr1e3(nnUNetTrainer):
#     def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
#                  device: torch.device = torch.device('cuda')):
#         # From https://grugbrain.dev/. Worth a read ya big brains ;-)
#
#         # apex predator of grug is complexity
#         # complexity bad
#         # say again:
#         # complexity very bad
#         # you say now:
#         # complexity very, very bad
#         # given choice between complexity or one on one against t-rex, grug take t-rex: at least grug see t-rex
#         # complexity is spirit demon that enter codebase through well-meaning but ultimately very clubbable non grug-brain developers and project managers who not fear complexity spirit demon or even know about sometime
#         # one day code base understandable and grug can get work done, everything good!
#         # next day impossible: complexity demon spirit has entered code and very dangerous situation!
#
#         # OK OK I am guilty. But I tried.
#         # https://www.osnews.com/images/comics/wtfm.jpg
#         # https://i.pinimg.com/originals/26/b2/50/26b250a738ea4abc7a5af4d42ad93af0.jpg
#
#         self.is_ddp = dist.is_available() and dist.is_initialized()
#         self.local_rank = 0 if not self.is_ddp else dist.get_rank()
#
#         self.device = device
#
#         # print what device we are using
#         if self.is_ddp:  # implicitly it's clear that we use cuda in this case
#             print(f"I am local rank {self.local_rank}. {device_count()} GPUs are available. The world size is "
#                   f"{dist.get_world_size()}."
#                   f"Setting device to {self.device}")
#             self.device = torch.device(type='cuda', index=self.local_rank)
#         else:
#             if self.device.type == 'cuda':
#                 # we might want to let the user pick this but for now please pick the correct GPU with CUDA_VISIBLE_DEVICES=X
#                 self.device = torch.device(type='cuda', index=0)
#             print(f"Using device: {self.device}")
#
#         # loading and saving this class for continuing from checkpoint should not happen based on pickling. This
#         # would also pickle the network etc. Bad, bad. Instead we just reinstantiate and then load the checkpoint we
#         # need. So let's save the init args
#         self.my_init_kwargs = {}
#         for k in inspect.signature(self.__init__).parameters.keys():
#             self.my_init_kwargs[k] = locals()[k]
#
#         ###  Saving all the init args into class variables for later access
#         self.plans_manager = PlansManager(plans)
#         self.configuration_manager = self.plans_manager.get_configuration(configuration)
#         self.configuration_name = configuration
#         self.dataset_json = dataset_json
#         self.fold = fold
#         self.unpack_dataset = unpack_dataset
#
#         ### Setting all the folder names. We need to make sure things don't crash in case we are just running
#         # inference and some of the folders may not be defined!
#         self.preprocessed_dataset_folder_base = join(nnUNet_preprocessed, self.plans_manager.dataset_name) \
#             if nnUNet_preprocessed is not None else None
#         self.output_folder_base = join(nnUNet_results, self.plans_manager.dataset_name,
#                                        self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" + configuration) \
#             if nnUNet_results is not None else None
#         if 'pretrain_name' in plans.keys():
#             pretrain_name = plans['pretrain_name']
#         else:
#             pretrain_name= None
#         if pretrain_name is not None:
#             self.output_folder_base = join(nnUNet_results, self.plans_manager.dataset_name, 'pretrained', pretrain_name,
#                                            self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" + configuration) \
#                 if nnUNet_results is not None else None
#             self.output_folder = join(self.output_folder_base, f'fold_{fold}')
#
#         else:
#             raise RuntimeError('No pretraining weights provided. Makes no sense to use this trainier')
#
#         self.preprocessed_dataset_folder = join(self.preprocessed_dataset_folder_base,
#                                                 self.configuration_manager.data_identifier)
#         # unlike the previous nnunet folder_with_segs_from_previous_stage is now part of the plans. For now it has to
#         # be a different configuration in the same plans
#         # IMPORTANT! the mapping must be bijective, so lowres must point to fullres and vice versa (using
#         # "previous_stage" and "next_stage"). Otherwise it won't work!
#         self.is_cascaded = self.configuration_manager.previous_stage_name is not None
#         self.folder_with_segs_from_previous_stage = \
#             join(nnUNet_results, self.plans_manager.dataset_name,
#                  self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" +
#                  self.configuration_manager.previous_stage_name, 'predicted_next_stage', self.configuration_name) \
#                 if self.is_cascaded else None
#
#         ### Some hyperparameters for you to fiddle with
#         self.initial_lr = 1e-3
#         self.weight_decay = 3e-5
#         self.oversample_foreground_percent = 0.33
#         self.num_iterations_per_epoch = 250
#         self.num_val_iterations_per_epoch = 50
#         self.num_epochs = 1000
#         self.current_epoch = 0
#         self.enable_deep_supervision = True
#
#
#         ### Dealing with labels/regions
#         self.label_manager = self.plans_manager.get_label_manager(dataset_json)
#         # labels can either be a list of int (regular training) or a list of tuples of int (region-based training)
#         # needed for predictions. We do sigmoid in case of (overlapping) regions
#
#         self.num_input_channels = None  # -> self.initialize()
#         self.network = None  # -> self.build_network_architecture()
#         self.optimizer = self.lr_scheduler = None  # -> self.initialize
#         self.grad_scaler = GradScaler() if self.device.type == 'cuda' else None
#         self.loss = None  # -> self.initialize
#
#         ### Simple logging. Don't take that away from me!
#         # initialize log file. This is just our log for the print statements etc. Not to be confused with lightning
#         # logging
#         timestamp = datetime.now()
#         maybe_mkdir_p(self.output_folder)
#         self.log_file = join(self.output_folder, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
#                              (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
#                               timestamp.second))
#         self.logger = nnUNetLogger()
#
#         ### placeholders
#         self.dataloader_train = self.dataloader_val = None  # see on_train_start
#
#         ### initializing stuff for remembering things and such
#         self._best_ema = None
#
#         ### inference things
#         self.inference_allowed_mirroring_axes = None  # this variable is set in
#         # self.configure_rotation_dummyDA_mirroring_and_inital_patch_size and will be saved in checkpoints
#
#         ### checkpoint saving stuff
#         self.save_every = 50
#         self.disable_checkpointing = False
#
#         ## DDP batch size and oversampling can differ between workers and needs adaptation
#         # we need to change the batch size in DDP because we don't use any of those distributed samplers
#         self._set_batch_size_and_oversample()
#
#         self.was_initialized = False
#
#         self.print_to_log_file("\n#######################################################################\n"
#                                "Please cite the following paper when using nnU-Net:\n"
#                                "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
#                                "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
#                                "Nature methods, 18(2), 203-211.\n"
#                                "#######################################################################\n",
#                                also_print_to_console=True, add_timestamp=False)
#
# class nnUNetTrainer_lr1e3_nosmooth(nnUNetTrainerDiceCELoss_noSmooth):
#     def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
#                  device: torch.device = torch.device('cuda')):
#         # From https://grugbrain.dev/. Worth a read ya big brains ;-)
#
#         # apex predator of grug is complexity
#         # complexity bad
#         # say again:
#         # complexity very bad
#         # you say now:
#         # complexity very, very bad
#         # given choice between complexity or one on one against t-rex, grug take t-rex: at least grug see t-rex
#         # complexity is spirit demon that enter codebase through well-meaning but ultimately very clubbable non grug-brain developers and project managers who not fear complexity spirit demon or even know about sometime
#         # one day code base understandable and grug can get work done, everything good!
#         # next day impossible: complexity demon spirit has entered code and very dangerous situation!
#
#         # OK OK I am guilty. But I tried.
#         # https://www.osnews.com/images/comics/wtfm.jpg
#         # https://i.pinimg.com/originals/26/b2/50/26b250a738ea4abc7a5af4d42ad93af0.jpg
#
#         self.is_ddp = dist.is_available() and dist.is_initialized()
#         self.local_rank = 0 if not self.is_ddp else dist.get_rank()
#
#         self.device = device
#
#         # print what device we are using
#         if self.is_ddp:  # implicitly it's clear that we use cuda in this case
#             print(f"I am local rank {self.local_rank}. {device_count()} GPUs are available. The world size is "
#                   f"{dist.get_world_size()}."
#                   f"Setting device to {self.device}")
#             self.device = torch.device(type='cuda', index=self.local_rank)
#         else:
#             if self.device.type == 'cuda':
#                 # we might want to let the user pick this but for now please pick the correct GPU with CUDA_VISIBLE_DEVICES=X
#                 self.device = torch.device(type='cuda', index=0)
#             print(f"Using device: {self.device}")
#
#         # loading and saving this class for continuing from checkpoint should not happen based on pickling. This
#         # would also pickle the network etc. Bad, bad. Instead we just reinstantiate and then load the checkpoint we
#         # need. So let's save the init args
#         self.my_init_kwargs = {}
#         for k in inspect.signature(self.__init__).parameters.keys():
#             self.my_init_kwargs[k] = locals()[k]
#
#         ###  Saving all the init args into class variables for later access
#         self.plans_manager = PlansManager(plans)
#         self.configuration_manager = self.plans_manager.get_configuration(configuration)
#         self.configuration_name = configuration
#         self.dataset_json = dataset_json
#         self.fold = fold
#         self.unpack_dataset = unpack_dataset
#
#         ### Setting all the folder names. We need to make sure things don't crash in case we are just running
#         # inference and some of the folders may not be defined!
#         self.preprocessed_dataset_folder_base = join(nnUNet_preprocessed, self.plans_manager.dataset_name) \
#             if nnUNet_preprocessed is not None else None
#         self.output_folder_base = join(nnUNet_results, self.plans_manager.dataset_name,
#                                        self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" + configuration) \
#             if nnUNet_results is not None else None
#         if 'pretrain_name' in plans.keys():
#             pretrain_name = plans['pretrain_name']
#         else:
#             pretrain_name= None
#         if pretrain_name is not None:
#             self.output_folder_base = join(nnUNet_results, self.plans_manager.dataset_name, 'pretrained', pretrain_name,
#                                            self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" + configuration) \
#                 if nnUNet_results is not None else None
#             self.output_folder = join(self.output_folder_base, f'fold_{fold}')
#
#         else:
#             raise RuntimeError('No pretraining weights provided. Makes no sense to use this trainier')
#
#         self.preprocessed_dataset_folder = join(self.preprocessed_dataset_folder_base,
#                                                 self.configuration_manager.data_identifier)
#         # unlike the previous nnunet folder_with_segs_from_previous_stage is now part of the plans. For now it has to
#         # be a different configuration in the same plans
#         # IMPORTANT! the mapping must be bijective, so lowres must point to fullres and vice versa (using
#         # "previous_stage" and "next_stage"). Otherwise it won't work!
#         self.is_cascaded = self.configuration_manager.previous_stage_name is not None
#         self.folder_with_segs_from_previous_stage = \
#             join(nnUNet_results, self.plans_manager.dataset_name,
#                  self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" +
#                  self.configuration_manager.previous_stage_name, 'predicted_next_stage', self.configuration_name) \
#                 if self.is_cascaded else None
#
#         ### Some hyperparameters for you to fiddle with
#         self.initial_lr = 1e-3
#         self.weight_decay = 3e-5
#         self.oversample_foreground_percent = 0.33
#         self.num_iterations_per_epoch = 250
#         self.num_val_iterations_per_epoch = 50
#         self.num_epochs = 1000
#         self.current_epoch = 0
#         self.enable_deep_supervision = True
#
#
#         ### Dealing with labels/regions
#         self.label_manager = self.plans_manager.get_label_manager(dataset_json)
#         # labels can either be a list of int (regular training) or a list of tuples of int (region-based training)
#         # needed for predictions. We do sigmoid in case of (overlapping) regions
#
#         self.num_input_channels = None  # -> self.initialize()
#         self.network = None  # -> self.build_network_architecture()
#         self.optimizer = self.lr_scheduler = None  # -> self.initialize
#         self.grad_scaler = GradScaler() if self.device.type == 'cuda' else None
#         self.loss = None  # -> self.initialize
#
#         ### Simple logging. Don't take that away from me!
#         # initialize log file. This is just our log for the print statements etc. Not to be confused with lightning
#         # logging
#         timestamp = datetime.now()
#         maybe_mkdir_p(self.output_folder)
#         self.log_file = join(self.output_folder, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
#                              (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
#                               timestamp.second))
#         self.logger = nnUNetLogger()
#
#         ### placeholders
#         self.dataloader_train = self.dataloader_val = None  # see on_train_start
#
#         ### initializing stuff for remembering things and such
#         self._best_ema = None
#
#         ### inference things
#         self.inference_allowed_mirroring_axes = None  # this variable is set in
#         # self.configure_rotation_dummyDA_mirroring_and_inital_patch_size and will be saved in checkpoints
#
#         ### checkpoint saving stuff
#         self.save_every = 50
#         self.disable_checkpointing = False
#
#         ## DDP batch size and oversampling can differ between workers and needs adaptation
#         # we need to change the batch size in DDP because we don't use any of those distributed samplers
#         self._set_batch_size_and_oversample()
#
#         self.was_initialized = False
#
#         self.print_to_log_file("\n#######################################################################\n"
#                                "Please cite the following paper when using nnU-Net:\n"
#                                "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
#                                "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
#                                "Nature methods, 18(2), 203-211.\n"
#                                "#######################################################################\n",
#                                also_print_to_console=True, add_timestamp=False)
# class nnUNetTrainer_pretrained(nnUNetTrainer):
#     def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
#                  device: torch.device = torch.device('cuda')):
#         # From https://grugbrain.dev/. Worth a read ya big brains ;-)
#
#         # apex predator of grug is complexity
#         # complexity bad
#         # say again:
#         # complexity very bad
#         # you say now:
#         # complexity very, very bad
#         # given choice between complexity or one on one against t-rex, grug take t-rex: at least grug see t-rex
#         # complexity is spirit demon that enter codebase through well-meaning but ultimately very clubbable non grug-brain developers and project managers who not fear complexity spirit demon or even know about sometime
#         # one day code base understandable and grug can get work done, everything good!
#         # next day impossible: complexity demon spirit has entered code and very dangerous situation!
#
#         # OK OK I am guilty. But I tried.
#         # https://www.osnews.com/images/comics/wtfm.jpg
#         # https://i.pinimg.com/originals/26/b2/50/26b250a738ea4abc7a5af4d42ad93af0.jpg
#
#         self.is_ddp = dist.is_available() and dist.is_initialized()
#         self.local_rank = 0 if not self.is_ddp else dist.get_rank()
#
#         self.device = device
#
#         # print what device we are using
#         if self.is_ddp:  # implicitly it's clear that we use cuda in this case
#             print(f"I am local rank {self.local_rank}. {device_count()} GPUs are available. The world size is "
#                   f"{dist.get_world_size()}."
#                   f"Setting device to {self.device}")
#             self.device = torch.device(type='cuda', index=self.local_rank)
#         else:
#             if self.device.type == 'cuda':
#                 # we might want to let the user pick this but for now please pick the correct GPU with CUDA_VISIBLE_DEVICES=X
#                 self.device = torch.device(type='cuda', index=0)
#             print(f"Using device: {self.device}")
#
#         # loading and saving this class for continuing from checkpoint should not happen based on pickling. This
#         # would also pickle the network etc. Bad, bad. Instead we just reinstantiate and then load the checkpoint we
#         # need. So let's save the init args
#         self.my_init_kwargs = {}
#         for k in inspect.signature(self.__init__).parameters.keys():
#             self.my_init_kwargs[k] = locals()[k]
#
#         ###  Saving all the init args into class variables for later access
#         self.plans_manager = PlansManager(plans)
#         self.configuration_manager = self.plans_manager.get_configuration(configuration)
#         self.configuration_name = configuration
#         self.dataset_json = dataset_json
#         self.fold = fold
#         self.unpack_dataset = unpack_dataset
#
#         ### Setting all the folder names. We need to make sure things don't crash in case we are just running
#         # inference and some of the folders may not be defined!
#         self.preprocessed_dataset_folder_base = join(nnUNet_preprocessed, self.plans_manager.dataset_name) \
#             if nnUNet_preprocessed is not None else None
#         self.output_folder_base = join(nnUNet_results, self.plans_manager.dataset_name,
#                                        self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" + configuration) \
#             if nnUNet_results is not None else None
#         if 'pretrain_name' in plans.keys():
#             pretrain_name = plans['pretrain_name']
#         else:
#             pretrain_name= None
#         if pretrain_name is not None:
#             self.output_folder_base = join(nnUNet_results, self.plans_manager.dataset_name, 'pretrained', pretrain_name,
#                                            self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" + configuration) \
#                 if nnUNet_results is not None else None
#             self.output_folder = join(self.output_folder_base, f'fold_{fold}')
#
#         else:
#             raise RuntimeError('No pretraining weights provided. Makes no sense to use this trainier')
#
#         self.preprocessed_dataset_folder = join(self.preprocessed_dataset_folder_base,
#                                                 self.configuration_manager.data_identifier)
#         # unlike the previous nnunet folder_with_segs_from_previous_stage is now part of the plans. For now it has to
#         # be a different configuration in the same plans
#         # IMPORTANT! the mapping must be bijective, so lowres must point to fullres and vice versa (using
#         # "previous_stage" and "next_stage"). Otherwise it won't work!
#         self.is_cascaded = self.configuration_manager.previous_stage_name is not None
#         self.folder_with_segs_from_previous_stage = \
#             join(nnUNet_results, self.plans_manager.dataset_name,
#                  self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" +
#                  self.configuration_manager.previous_stage_name, 'predicted_next_stage', self.configuration_name) \
#                 if self.is_cascaded else None
#
#         ### Some hyperparameters for you to fiddle with
#         self.initial_lr = 1e-2
#         self.weight_decay = 3e-5
#         self.oversample_foreground_percent = 0.33
#         self.num_iterations_per_epoch = 250
#         self.num_val_iterations_per_epoch = 50
#         self.num_epochs = 1000
#         self.current_epoch = 0
#         self.enable_deep_supervision = True
#
#
#         ### Dealing with labels/regions
#         self.label_manager = self.plans_manager.get_label_manager(dataset_json)
#         # labels can either be a list of int (regular training) or a list of tuples of int (region-based training)
#         # needed for predictions. We do sigmoid in case of (overlapping) regions
#
#         self.num_input_channels = None  # -> self.initialize()
#         self.network = None  # -> self.build_network_architecture()
#         self.optimizer = self.lr_scheduler = None  # -> self.initialize
#         self.grad_scaler = GradScaler() if self.device.type == 'cuda' else None
#         self.loss = None  # -> self.initialize
#
#         ### Simple logging. Don't take that away from me!
#         # initialize log file. This is just our log for the print statements etc. Not to be confused with lightning
#         # logging
#         timestamp = datetime.now()
#         maybe_mkdir_p(self.output_folder)
#         self.log_file = join(self.output_folder, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
#                              (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
#                               timestamp.second))
#         self.logger = nnUNetLogger()
#
#         ### placeholders
#         self.dataloader_train = self.dataloader_val = None  # see on_train_start
#
#         ### initializing stuff for remembering things and such
#         self._best_ema = None
#
#         ### inference things
#         self.inference_allowed_mirroring_axes = None  # this variable is set in
#         # self.configure_rotation_dummyDA_mirroring_and_inital_patch_size and will be saved in checkpoints
#
#         ### checkpoint saving stuff
#         self.save_every = 50
#         self.disable_checkpointing = False
#
#         ## DDP batch size and oversampling can differ between workers and needs adaptation
#         # we need to change the batch size in DDP because we don't use any of those distributed samplers
#         self._set_batch_size_and_oversample()
#
#         self.was_initialized = False
#
#         self.print_to_log_file("\n#######################################################################\n"
#                                "Please cite the following paper when using nnU-Net:\n"
#                                "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
#                                "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
#                                "Nature methods, 18(2), 203-211.\n"
#                                "#######################################################################\n",
#                                also_print_to_console=True, add_timestamp=False)
#
# class nnUNetTrainer_pretrained_encoder(nnUNetTrainer):
#     def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
#                  device: torch.device = torch.device('cuda')):
#         # From https://grugbrain.dev/. Worth a read ya big brains ;-)
#
#         # apex predator of grug is complexity
#         # complexity bad
#         # say again:
#         # complexity very bad
#         # you say now:
#         # complexity very, very bad
#         # given choice between complexity or one on one against t-rex, grug take t-rex: at least grug see t-rex
#         # complexity is spirit demon that enter codebase through well-meaning but ultimately very clubbable non grug-brain developers and project managers who not fear complexity spirit demon or even know about sometime
#         # one day code base understandable and grug can get work done, everything good!
#         # next day impossible: complexity demon spirit has entered code and very dangerous situation!
#
#         # OK OK I am guilty. But I tried.
#         # https://www.osnews.com/images/comics/wtfm.jpg
#         # https://i.pinimg.com/originals/26/b2/50/26b250a738ea4abc7a5af4d42ad93af0.jpg
#
#         self.is_ddp = dist.is_available() and dist.is_initialized()
#         self.local_rank = 0 if not self.is_ddp else dist.get_rank()
#
#         self.device = device
#
#         # print what device we are using
#         if self.is_ddp:  # implicitly it's clear that we use cuda in this case
#             print(f"I am local rank {self.local_rank}. {device_count()} GPUs are available. The world size is "
#                   f"{dist.get_world_size()}."
#                   f"Setting device to {self.device}")
#             self.device = torch.device(type='cuda', index=self.local_rank)
#         else:
#             if self.device.type == 'cuda':
#                 # we might want to let the user pick this but for now please pick the correct GPU with CUDA_VISIBLE_DEVICES=X
#                 self.device = torch.device(type='cuda', index=0)
#             print(f"Using device: {self.device}")
#
#         # loading and saving this class for continuing from checkpoint should not happen based on pickling. This
#         # would also pickle the network etc. Bad, bad. Instead we just reinstantiate and then load the checkpoint we
#         # need. So let's save the init args
#         self.my_init_kwargs = {}
#         for k in inspect.signature(self.__init__).parameters.keys():
#             self.my_init_kwargs[k] = locals()[k]
#
#         ###  Saving all the init args into class variables for later access
#         self.plans_manager = PlansManager(plans)
#         self.configuration_manager = self.plans_manager.get_configuration(configuration)
#         self.configuration_name = configuration
#         self.dataset_json = dataset_json
#         self.fold = fold
#         self.unpack_dataset = unpack_dataset
#
#         ### Setting all the folder names. We need to make sure things don't crash in case we are just running
#         # inference and some of the folders may not be defined!
#         self.preprocessed_dataset_folder_base = join(nnUNet_preprocessed, self.plans_manager.dataset_name) \
#             if nnUNet_preprocessed is not None else None
#         self.output_folder_base = join(nnUNet_results, self.plans_manager.dataset_name,
#                                        self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" + configuration) \
#             if nnUNet_results is not None else None
#         if 'pretrain_name' in plans.keys():
#             pretrain_name = plans['pretrain_name']
#         else:
#             pretrain_name= None
#         if pretrain_name is not None:
#             self.output_folder_base = join(nnUNet_results, self.plans_manager.dataset_name, 'pretrained', pretrain_name,
#                                            self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" + configuration) \
#                 if nnUNet_results is not None else None
#             self.output_folder = join(self.output_folder_base, f'fold_{fold}')
#
#         else:
#             raise RuntimeError('No pretraining weights provided. Makes no sense to use this trainier')
#
#         self.preprocessed_dataset_folder = join(self.preprocessed_dataset_folder_base,
#                                                 self.configuration_manager.data_identifier)
#         # unlike the previous nnunet folder_with_segs_from_previous_stage is now part of the plans. For now it has to
#         # be a different configuration in the same plans
#         # IMPORTANT! the mapping must be bijective, so lowres must point to fullres and vice versa (using
#         # "previous_stage" and "next_stage"). Otherwise it won't work!
#         self.is_cascaded = self.configuration_manager.previous_stage_name is not None
#         self.folder_with_segs_from_previous_stage = \
#             join(nnUNet_results, self.plans_manager.dataset_name,
#                  self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" +
#                  self.configuration_manager.previous_stage_name, 'predicted_next_stage', self.configuration_name) \
#                 if self.is_cascaded else None
#
#         ### Some hyperparameters for you to fiddle with
#         self.initial_lr = 1e-2
#         self.weight_decay = 3e-5
#         self.oversample_foreground_percent = 0.33
#         self.num_iterations_per_epoch = 250
#         self.num_val_iterations_per_epoch = 50
#         self.num_epochs = 1000
#         self.current_epoch = 0
#         self.enable_deep_supervision = True
#
#
#         ### Dealing with labels/regions
#         self.label_manager = self.plans_manager.get_label_manager(dataset_json)
#         # labels can either be a list of int (regular training) or a list of tuples of int (region-based training)
#         # needed for predictions. We do sigmoid in case of (overlapping) regions
#
#         self.num_input_channels = None  # -> self.initialize()
#         self.network = None  # -> self.build_network_architecture()
#         self.optimizer = self.lr_scheduler = None  # -> self.initialize
#         self.grad_scaler = GradScaler() if self.device.type == 'cuda' else None
#         self.loss = None  # -> self.initialize
#
#         ### Simple logging. Don't take that away from me!
#         # initialize log file. This is just our log for the print statements etc. Not to be confused with lightning
#         # logging
#         timestamp = datetime.now()
#         maybe_mkdir_p(self.output_folder)
#         self.log_file = join(self.output_folder, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
#                              (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
#                               timestamp.second))
#         self.logger = nnUNetLogger()
#
#         ### placeholders
#         self.dataloader_train = self.dataloader_val = None  # see on_train_start
#
#         ### initializing stuff for remembering things and such
#         self._best_ema = None
#
#         ### inference things
#         self.inference_allowed_mirroring_axes = None  # this variable is set in
#         # self.configure_rotation_dummyDA_mirroring_and_inital_patch_size and will be saved in checkpoints
#
#         ### checkpoint saving stuff
#         self.save_every = 50
#         self.disable_checkpointing = False
#
#         ## DDP batch size and oversampling can differ between workers and needs adaptation
#         # we need to change the batch size in DDP because we don't use any of those distributed samplers
#         self._set_batch_size_and_oversample()
#
#         self.was_initialized = False
#
#         self.print_to_log_file("\n#######################################################################\n"
#                                "Please cite the following paper when using nnU-Net:\n"
#                                "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
#                                "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
#                                "Nature methods, 18(2), 203-211.\n"
#                                "#######################################################################\n",
#                                also_print_to_console=True, add_timestamp=False)
# class nnUNetTrainer_warmupdecoder(nnUNetTrainer):
#     def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
#                  device: torch.device = torch.device('cuda')):
#         # From https://grugbrain.dev/. Worth a read ya big brains ;-)
#
#         # apex predator of grug is complexity
#         # complexity bad
#         # say again:
#         # complexity very bad
#         # you say now:
#         # complexity very, very bad
#         # given choice between complexity or one on one against t-rex, grug take t-rex: at least grug see t-rex
#         # complexity is spirit demon that enter codebase through well-meaning but ultimately very clubbable non grug-brain developers and project managers who not fear complexity spirit demon or even know about sometime
#         # one day code base understandable and grug can get work done, everything good!
#         # next day impossible: complexity demon spirit has entered code and very dangerous situation!
#
#         # OK OK I am guilty. But I tried.
#         # https://www.osnews.com/images/comics/wtfm.jpg
#         # https://i.pinimg.com/originals/26/b2/50/26b250a738ea4abc7a5af4d42ad93af0.jpg
#
#         self.is_ddp = dist.is_available() and dist.is_initialized()
#         self.local_rank = 0 if not self.is_ddp else dist.get_rank()
#
#         self.device = device
#
#         # print what device we are using
#         if self.is_ddp:  # implicitly it's clear that we use cuda in this case
#             print(f"I am local rank {self.local_rank}. {device_count()} GPUs are available. The world size is "
#                   f"{dist.get_world_size()}."
#                   f"Setting device to {self.device}")
#             self.device = torch.device(type='cuda', index=self.local_rank)
#         else:
#             if self.device.type == 'cuda':
#                 # we might want to let the user pick this but for now please pick the correct GPU with CUDA_VISIBLE_DEVICES=X
#                 self.device = torch.device(type='cuda', index=0)
#             print(f"Using device: {self.device}")
#
#         # loading and saving this class for continuing from checkpoint should not happen based on pickling. This
#         # would also pickle the network etc. Bad, bad. Instead we just reinstantiate and then load the checkpoint we
#         # need. So let's save the init args
#         self.my_init_kwargs = {}
#         for k in inspect.signature(self.__init__).parameters.keys():
#             self.my_init_kwargs[k] = locals()[k]
#
#         ###  Saving all the init args into class variables for later access
#         self.plans_manager = PlansManager(plans)
#         self.configuration_manager = self.plans_manager.get_configuration(configuration)
#         self.configuration_name = configuration
#         self.dataset_json = dataset_json
#         self.fold = fold
#         self.unpack_dataset = unpack_dataset
#
#         ### Setting all the folder names. We need to make sure things don't crash in case we are just running
#         # inference and some of the folders may not be defined!
#         self.preprocessed_dataset_folder_base = join(nnUNet_preprocessed, self.plans_manager.dataset_name) \
#             if nnUNet_preprocessed is not None else None
#         self.output_folder_base = join(nnUNet_results, self.plans_manager.dataset_name,
#                                        self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" + configuration) \
#             if nnUNet_results is not None else None
#         if 'pretrain_name' in plans.keys():
#             pretrain_name = plans['pretrain_name']
#         else:
#             pretrain_name= None
#         if pretrain_name is not None:
#             self.output_folder_base = join(nnUNet_results, self.plans_manager.dataset_name, 'pretrained', pretrain_name,
#                                            self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" + configuration) \
#                 if nnUNet_results is not None else None
#             self.output_folder = join(self.output_folder_base, f'fold_{fold}')
#
#         else:
#             raise RuntimeError('No pretraining weights provided. Makes no sense to use this trainier')
#
#         self.preprocessed_dataset_folder = join(self.preprocessed_dataset_folder_base,
#                                                 self.configuration_manager.data_identifier)
#         # unlike the previous nnunet folder_with_segs_from_previous_stage is now part of the plans. For now it has to
#         # be a different configuration in the same plans
#         # IMPORTANT! the mapping must be bijective, so lowres must point to fullres and vice versa (using
#         # "previous_stage" and "next_stage"). Otherwise it won't work!
#         self.is_cascaded = self.configuration_manager.previous_stage_name is not None
#         self.folder_with_segs_from_previous_stage = \
#             join(nnUNet_results, self.plans_manager.dataset_name,
#                  self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" +
#                  self.configuration_manager.previous_stage_name, 'predicted_next_stage', self.configuration_name) \
#                 if self.is_cascaded else None
#
#         ### Some hyperparameters for you to fiddle with
#         self.initial_lr = 1e-2
#         self.weight_decay = 3e-5
#         self.oversample_foreground_percent = 0.33
#         self.num_iterations_per_epoch = 250
#         self.num_val_iterations_per_epoch = 50
#         self.num_epochs = 1000
#         self.current_epoch = 0
#         self.enable_deep_supervision = True
#
#         #### hyperparameter for warmup
#         self.warmup_duration_whole_net = 50  # lin increase whole network
#         self.warmup_heads_max_lr = 1e-3  # lin increase lr for heads
#         self.warmup_duration_heads = 50  # this is for the DECODER
#         self.num_epochs = 1000 + self.warmup_duration_whole_net + self.warmup_duration_heads
#
#         ### Dealing with labels/regions
#         self.label_manager = self.plans_manager.get_label_manager(dataset_json)
#         # labels can either be a list of int (regular training) or a list of tuples of int (region-based training)
#         # needed for predictions. We do sigmoid in case of (overlapping) regions
#
#         self.num_input_channels = None  # -> self.initialize()
#         self.network = None  # -> self.build_network_architecture()
#         self.optimizer = self.lr_scheduler = None  # -> self.initialize
#         self.grad_scaler = GradScaler() if self.device.type == 'cuda' else None
#         self.loss = None  # -> self.initialize
#
#         ### Simple logging. Don't take that away from me!
#         # initialize log file. This is just our log for the print statements etc. Not to be confused with lightning
#         # logging
#         timestamp = datetime.now()
#         maybe_mkdir_p(self.output_folder)
#         self.log_file = join(self.output_folder, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
#                              (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
#                               timestamp.second))
#         self.logger = nnUNetLogger()
#
#         ### placeholders
#         self.dataloader_train = self.dataloader_val = None  # see on_train_start
#
#         ### initializing stuff for remembering things and such
#         self._best_ema = None
#
#         ### inference things
#         self.inference_allowed_mirroring_axes = None  # this variable is set in
#         # self.configure_rotation_dummyDA_mirroring_and_inital_patch_size and will be saved in checkpoints
#
#         ### checkpoint saving stuff
#         self.save_every = 50
#         self.disable_checkpointing = False
#
#         ## DDP batch size and oversampling can differ between workers and needs adaptation
#         # we need to change the batch size in DDP because we don't use any of those distributed samplers
#         self._set_batch_size_and_oversample()
#
#         self.was_initialized = False
#
#         self.print_to_log_file("\n#######################################################################\n"
#                                "Please cite the following paper when using nnU-Net:\n"
#                                "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
#                                "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
#                                "Nature methods, 18(2), 203-211.\n"
#                                "#######################################################################\n",
#                                also_print_to_console=True, add_timestamp=False)
#
#     def initialize(self):
#         if not self.was_initialized:
#             self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
#                                                                    self.dataset_json)
#
#             self.network = self.build_network_architecture(
#                 self.configuration_manager.network_arch_class_name,
#                 self.configuration_manager.network_arch_init_kwargs,
#                 self.configuration_manager.network_arch_init_kwargs_req_import,
#                 self.num_input_channels,
#                 self.label_manager.num_segmentation_heads,
#                 self.enable_deep_supervision
#             ).to(self.device)
#             # compile network for free speedup
#             if self._do_i_compile():
#                 self.print_to_log_file('Using torch.compile...')
#                 self.network = torch.compile(self.network)
#
#             self.optimizer, self.lr_scheduler = self.configure_optimizers(True)
#             # if ddp, wrap in DDP wrapper
#             if self.is_ddp:
#                 self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
#                 self.network = DDP(self.network, device_ids=[self.local_rank])
#
#             self.loss = self._build_loss()
#             # torch 2.2.2 crashes upon compiling CE loss
#             # if self._do_i_compile():
#             #     self.loss = torch.compile(self.loss)
#             self.was_initialized = True
#         else:
#             raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
#                                "That should not happen.")
#
#     def configure_optimizers(self, dec_only=False, netwarmup=False):
#
#         if isinstance(self.network, DDP):
#             params = self.network.module.parameters()
#             dec = self.network.module.decoder.parameters()
#         else:
#             params = self.network.parameters()
#             # print(self.network.state_dict().keys())
#             dec = self.network.decoder.parameters()
#         if dec_only:
#             self.print_to_log_file("train only decoder")
#             optimizer = torch.optim.SGD(list(dec), self.initial_lr, weight_decay=self.weight_decay,
#                                         momentum=0.99, nesterov=True)
#             lr_scheduler = Lin_incr_LRScheduler(optimizer, self.warmup_heads_max_lr, self.warmup_duration_heads)
#         elif netwarmup and not dec_only:
#             self.print_to_log_file("train whole net, warmup")
#             optimizer = torch.optim.SGD(params, self.initial_lr, weight_decay=self.weight_decay,
#                                         momentum=0.99, nesterov=True)
#             lr_scheduler = Lin_incr_offset_LRScheduler(optimizer, self.initial_lr, self.warmup_duration_whole_net, self.warmup_duration_heads)
#         else:
#             self.print_to_log_file("train whole net, default schedule")
#             optimizer = torch.optim.SGD(params, self.initial_lr, weight_decay=self.weight_decay,
#                                         momentum=0.99, nesterov=True)
#             lr_scheduler = PolyLRScheduler_offset(optimizer, self.initial_lr, self.num_epochs, self.warmup_duration_whole_net + self.warmup_duration_heads)
#         return optimizer, lr_scheduler
#
#     def on_train_epoch_start(self):
#         self.network.train()
#         if self.current_epoch == self.warmup_duration_heads:
#             self.optimizer, self.lr_scheduler = self.configure_optimizers(False, True)
#         if self.current_epoch == self.warmup_duration_heads + self.warmup_duration_whole_net:
#             self.optimizer, self.lr_scheduler = self.configure_optimizers(False, False)
#         self.lr_scheduler.step(self.current_epoch)
#         self.print_to_log_file('')
#         self.print_to_log_file(f'Epoch {self.current_epoch}')
#         self.print_to_log_file(
#             f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}")
#         # lrs are the same for all workers so we don't need to gather them in case of DDP training
#         self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)
#
#     def on_epoch_start(self):
#         # if isinstance(self.network, DDP):
#         #     self.print_to_log_file('seg output weight', self.network.module.decoder.seg_layers[0].weight[0,0])
#         #     if self.has_stem:
#         #         self.print_to_log_file('stem weight', self.network.module.encoder.stem.convs[0].conv.weight[0, 0])
#         #         self.print_to_log_file('first fixed weight', self.network.module.encoder.stages[0].blocks[0].conv1.conv.weight[0, 0, 0])
#         #     else:
#         #         self.print_to_log_file('stem weight',self.network.module.encoder.stages[0][0].convs[0].conv.weight[0, 0, 0])
#         #         self.print_to_log_file('first fixed weight', self.network.module.encoder.stages[0][0].convs[1].conv.weight[0, 0, 0])
#         # else:
#         #     self.print_to_log_file('seg output weight', self.network.decoder.seg_layers[0].weight[0,0])
#         #     if self.has_stem:
#         #         self.print_to_log_file('stem weight', self.network.encoder.stem.convs[0].conv.weight[0, 0])
#         #         self.print_to_log_file('free weight', self.network.encoder.stages[0].blocks[0].conv1.conv.weight[0, 0, 0])
#         #     else:
#         #         self.print_to_log_file('stem weight',self.network.encoder.stages[0][0].convs[0].conv.weight[0, 0, 0])
#         #         self.print_to_log_file('first fixed', self.network.encoder.stages[0][0].convs[1].conv.weight[0, 0, 0])
#
#
#         self.logger.log('epoch_start_timestamps', time(), self.current_epoch)
#
#     def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
#         if not self.was_initialized:
#             self.initialize()
#
#         if isinstance(filename_or_checkpoint, str):
#             checkpoint = torch.load(filename_or_checkpoint, map_location=self.device)
#         # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
#         # match. Use heuristic to make it match
#         new_state_dict = {}
#         for k, value in checkpoint['network_weights'].items():
#             key = k
#             if key not in self.network.state_dict().keys() and key.startswith('module.'):
#                 key = key[7:]
#             new_state_dict[key] = value
#
#         self.my_init_kwargs = checkpoint['init_args']
#         self.current_epoch = checkpoint['current_epoch']
#         self.logger.load_checkpoint(checkpoint['logging'])
#         self._best_ema = checkpoint['_best_ema']
#         self.inference_allowed_mirroring_axes = checkpoint[
#             'inference_allowed_mirroring_axes'] if 'inference_allowed_mirroring_axes' in checkpoint.keys() else self.inference_allowed_mirroring_axes
#
#         # messing with state dict naming schemes. Facepalm.
#         if self.is_ddp:
#             if isinstance(self.network.module, OptimizedModule):
#                 self.network.module._orig_mod.load_state_dict(new_state_dict)
#             else:
#                 self.network.module.load_state_dict(new_state_dict)
#         else:
#             if isinstance(self.network, OptimizedModule):
#                 self.network._orig_mod.load_state_dict(new_state_dict)
#             else:
#                 self.network.load_state_dict(new_state_dict)
#
#         if self.warmup_duration_heads + self.warmup_duration_whole_net >= self.current_epoch >= self.warmup_duration_heads:
#             self.optimizer, self.lr_scheduler = self.configure_optimizers(False, True)
#         if self.current_epoch >= self.warmup_duration_heads + self.warmup_duration_whole_net:
#             self.optimizer, self.lr_scheduler = self.configure_optimizers(False, False)
#
#         self.optimizer.load_state_dict(checkpoint['optimizer_state'])
#
#
#
#         if self.grad_scaler is not None:
#             if checkpoint['grad_scaler_state'] is not None:
#                 self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])
#
#
#     def _build_loss(self):
#         # set smooth to 0
#         if self.label_manager.has_regions:
#             loss = DC_and_BCE_loss({},
#                                    {'batch_dice': self.configuration_manager.batch_dice,
#                                     'do_bg': True, 'smooth': 0, 'ddp': self.is_ddp},
#                                    use_ignore_label=self.label_manager.ignore_label is not None,
#                                    dice_class=MemoryEfficientSoftDiceLoss)
#         else:
#             loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
#                                    'smooth': 0, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
#                                   ignore_label=self.label_manager.ignore_label,
#                                   dice_class=MemoryEfficientSoftDiceLoss)
#
#         deep_supervision_scales = self._get_deep_supervision_scales()
#
#         # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
#         # this gives higher resolution outputs more weight in the loss
#         weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
#
#         # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
#         weights = weights / weights.sum()
#         # now wrap the loss
#         loss = DeepSupervisionWrapper(loss, weights)
#         return loss
#
# class nnUNetTrainer_warmupdecoder_nosmooth(nnUNetTrainerDiceCELoss_noSmooth):
#     def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
#                  device: torch.device = torch.device('cuda')):
#         # From https://grugbrain.dev/. Worth a read ya big brains ;-)
#
#         # apex predator of grug is complexity
#         # complexity bad
#         # say again:
#         # complexity very bad
#         # you say now:
#         # complexity very, very bad
#         # given choice between complexity or one on one against t-rex, grug take t-rex: at least grug see t-rex
#         # complexity is spirit demon that enter codebase through well-meaning but ultimately very clubbable non grug-brain developers and project managers who not fear complexity spirit demon or even know about sometime
#         # one day code base understandable and grug can get work done, everything good!
#         # next day impossible: complexity demon spirit has entered code and very dangerous situation!
#
#         # OK OK I am guilty. But I tried.
#         # https://www.osnews.com/images/comics/wtfm.jpg
#         # https://i.pinimg.com/originals/26/b2/50/26b250a738ea4abc7a5af4d42ad93af0.jpg
#
#         self.is_ddp = dist.is_available() and dist.is_initialized()
#         self.local_rank = 0 if not self.is_ddp else dist.get_rank()
#
#         self.device = device
#
#         # print what device we are using
#         if self.is_ddp:  # implicitly it's clear that we use cuda in this case
#             print(f"I am local rank {self.local_rank}. {device_count()} GPUs are available. The world size is "
#                   f"{dist.get_world_size()}."
#                   f"Setting device to {self.device}")
#             self.device = torch.device(type='cuda', index=self.local_rank)
#         else:
#             if self.device.type == 'cuda':
#                 # we might want to let the user pick this but for now please pick the correct GPU with CUDA_VISIBLE_DEVICES=X
#                 self.device = torch.device(type='cuda', index=0)
#             print(f"Using device: {self.device}")
#
#         # loading and saving this class for continuing from checkpoint should not happen based on pickling. This
#         # would also pickle the network etc. Bad, bad. Instead we just reinstantiate and then load the checkpoint we
#         # need. So let's save the init args
#         self.my_init_kwargs = {}
#         for k in inspect.signature(self.__init__).parameters.keys():
#             self.my_init_kwargs[k] = locals()[k]
#
#         ###  Saving all the init args into class variables for later access
#         self.plans_manager = PlansManager(plans)
#         self.configuration_manager = self.plans_manager.get_configuration(configuration)
#         self.configuration_name = configuration
#         self.dataset_json = dataset_json
#         self.fold = fold
#         self.unpack_dataset = unpack_dataset
#
#         ### Setting all the folder names. We need to make sure things don't crash in case we are just running
#         # inference and some of the folders may not be defined!
#         self.preprocessed_dataset_folder_base = join(nnUNet_preprocessed, self.plans_manager.dataset_name) \
#             if nnUNet_preprocessed is not None else None
#         self.output_folder_base = join(nnUNet_results, self.plans_manager.dataset_name,
#                                        self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" + configuration) \
#             if nnUNet_results is not None else None
#         if 'pretrain_name' in plans.keys():
#             pretrain_name = plans['pretrain_name']
#         else:
#             pretrain_name= None
#         if pretrain_name is not None:
#             self.output_folder_base = join(nnUNet_results, self.plans_manager.dataset_name, 'pretrained', pretrain_name,
#                                            self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" + configuration) \
#                 if nnUNet_results is not None else None
#             self.output_folder = join(self.output_folder_base, f'fold_{fold}')
#
#         else:
#             raise RuntimeError('No pretraining weights provided. Makes no sense to use this trainier')
#
#         self.preprocessed_dataset_folder = join(self.preprocessed_dataset_folder_base,
#                                                 self.configuration_manager.data_identifier)
#         # unlike the previous nnunet folder_with_segs_from_previous_stage is now part of the plans. For now it has to
#         # be a different configuration in the same plans
#         # IMPORTANT! the mapping must be bijective, so lowres must point to fullres and vice versa (using
#         # "previous_stage" and "next_stage"). Otherwise it won't work!
#         self.is_cascaded = self.configuration_manager.previous_stage_name is not None
#         self.folder_with_segs_from_previous_stage = \
#             join(nnUNet_results, self.plans_manager.dataset_name,
#                  self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" +
#                  self.configuration_manager.previous_stage_name, 'predicted_next_stage', self.configuration_name) \
#                 if self.is_cascaded else None
#
#         ### Some hyperparameters for you to fiddle with
#         self.initial_lr = 1e-2
#         self.weight_decay = 3e-5
#         self.oversample_foreground_percent = 0.33
#         self.num_iterations_per_epoch = 250
#         self.num_val_iterations_per_epoch = 50
#         self.num_epochs = 1000
#         self.current_epoch = 0
#         self.enable_deep_supervision = True
#
#         #### hyperparameter for warmup
#         self.warmup_duration_whole_net = 50  # lin increase whole network
#         self.warmup_heads_max_lr = 1e-3  # lin increase lr for heads
#         self.warmup_duration_heads = 50  # this is for the DECODER
#         self.num_epochs = 1000 + self.warmup_duration_whole_net + self.warmup_duration_heads
#
#         ### Dealing with labels/regions
#         self.label_manager = self.plans_manager.get_label_manager(dataset_json)
#         # labels can either be a list of int (regular training) or a list of tuples of int (region-based training)
#         # needed for predictions. We do sigmoid in case of (overlapping) regions
#
#         self.num_input_channels = None  # -> self.initialize()
#         self.network = None  # -> self.build_network_architecture()
#         self.optimizer = self.lr_scheduler = None  # -> self.initialize
#         self.grad_scaler = GradScaler() if self.device.type == 'cuda' else None
#         self.loss = None  # -> self.initialize
#
#         ### Simple logging. Don't take that away from me!
#         # initialize log file. This is just our log for the print statements etc. Not to be confused with lightning
#         # logging
#         timestamp = datetime.now()
#         maybe_mkdir_p(self.output_folder)
#         self.log_file = join(self.output_folder, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
#                              (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
#                               timestamp.second))
#         self.logger = nnUNetLogger()
#
#         ### placeholders
#         self.dataloader_train = self.dataloader_val = None  # see on_train_start
#
#         ### initializing stuff for remembering things and such
#         self._best_ema = None
#
#         ### inference things
#         self.inference_allowed_mirroring_axes = None  # this variable is set in
#         # self.configure_rotation_dummyDA_mirroring_and_inital_patch_size and will be saved in checkpoints
#
#         ### checkpoint saving stuff
#         self.save_every = 50
#         self.disable_checkpointing = False
#
#         ## DDP batch size and oversampling can differ between workers and needs adaptation
#         # we need to change the batch size in DDP because we don't use any of those distributed samplers
#         self._set_batch_size_and_oversample()
#
#         self.was_initialized = False
#
#         self.print_to_log_file("\n#######################################################################\n"
#                                "Please cite the following paper when using nnU-Net:\n"
#                                "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
#                                "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
#                                "Nature methods, 18(2), 203-211.\n"
#                                "#######################################################################\n",
#                                also_print_to_console=True, add_timestamp=False)
#
#     def initialize(self):
#         if not self.was_initialized:
#             self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
#                                                                    self.dataset_json)
#
#             self.network = self.build_network_architecture(
#                 self.configuration_manager.network_arch_class_name,
#                 self.configuration_manager.network_arch_init_kwargs,
#                 self.configuration_manager.network_arch_init_kwargs_req_import,
#                 self.num_input_channels,
#                 self.label_manager.num_segmentation_heads,
#                 self.enable_deep_supervision
#             ).to(self.device)
#             # compile network for free speedup
#             if self._do_i_compile():
#                 self.print_to_log_file('Using torch.compile...')
#                 self.network = torch.compile(self.network)
#
#             self.optimizer, self.lr_scheduler = self.configure_optimizers(True)
#             # if ddp, wrap in DDP wrapper
#             if self.is_ddp:
#                 self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
#                 self.network = DDP(self.network, device_ids=[self.local_rank])
#
#             self.loss = self._build_loss()
#             # torch 2.2.2 crashes upon compiling CE loss
#             # if self._do_i_compile():
#             #     self.loss = torch.compile(self.loss)
#             self.was_initialized = True
#         else:
#             raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
#                                "That should not happen.")
#
#     def configure_optimizers(self, dec_only=False, netwarmup=False):
#
#         if isinstance(self.network, DDP):
#             params = self.network.module.parameters()
#             dec = self.network.module.decoder.parameters()
#         else:
#             params = self.network.parameters()
#             # print(self.network.state_dict().keys())
#             dec = self.network.decoder.parameters()
#         if dec_only:
#             self.print_to_log_file("train only decoder")
#             optimizer = torch.optim.SGD(list(dec), self.initial_lr, weight_decay=self.weight_decay,
#                                         momentum=0.99, nesterov=True)
#             lr_scheduler = Lin_incr_LRScheduler(optimizer, self.warmup_heads_max_lr, self.warmup_duration_heads)
#         elif netwarmup and not dec_only:
#             self.print_to_log_file("train whole net, warmup")
#             optimizer = torch.optim.SGD(params, self.initial_lr, weight_decay=self.weight_decay,
#                                         momentum=0.99, nesterov=True)
#             lr_scheduler = Lin_incr_offset_LRScheduler(optimizer, self.initial_lr, self.warmup_duration_whole_net, self.warmup_duration_heads)
#         else:
#             self.print_to_log_file("train whole net, default schedule")
#             optimizer = torch.optim.SGD(params, self.initial_lr, weight_decay=self.weight_decay,
#                                         momentum=0.99, nesterov=True)
#             lr_scheduler = PolyLRScheduler_offset(optimizer, self.initial_lr, self.num_epochs, self.warmup_duration_whole_net + self.warmup_duration_heads)
#         return optimizer, lr_scheduler
#
#     def on_train_epoch_start(self):
#         self.network.train()
#         if self.current_epoch == self.warmup_duration_heads:
#             self.optimizer, self.lr_scheduler = self.configure_optimizers(False, True)
#         if self.current_epoch == self.warmup_duration_heads + self.warmup_duration_whole_net:
#             self.optimizer, self.lr_scheduler = self.configure_optimizers(False, False)
#         self.lr_scheduler.step(self.current_epoch)
#         self.print_to_log_file('')
#         self.print_to_log_file(f'Epoch {self.current_epoch}')
#         self.print_to_log_file(
#             f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}")
#         # lrs are the same for all workers so we don't need to gather them in case of DDP training
#         self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)
#
#     def on_epoch_start(self):
#         # if isinstance(self.network, DDP):
#         #     self.print_to_log_file('seg output weight', self.network.module.decoder.seg_layers[0].weight[0,0])
#         #     if self.has_stem:
#         #         self.print_to_log_file('stem weight', self.network.module.encoder.stem.convs[0].conv.weight[0, 0])
#         #         self.print_to_log_file('first fixed weight', self.network.module.encoder.stages[0].blocks[0].conv1.conv.weight[0, 0, 0])
#         #     else:
#         #         self.print_to_log_file('stem weight',self.network.module.encoder.stages[0][0].convs[0].conv.weight[0, 0, 0])
#         #         self.print_to_log_file('first fixed weight', self.network.module.encoder.stages[0][0].convs[1].conv.weight[0, 0, 0])
#         # else:
#         #     self.print_to_log_file('seg output weight', self.network.decoder.seg_layers[0].weight[0,0])
#         #     if self.has_stem:
#         #         self.print_to_log_file('stem weight', self.network.encoder.stem.convs[0].conv.weight[0, 0])
#         #         self.print_to_log_file('free weight', self.network.encoder.stages[0].blocks[0].conv1.conv.weight[0, 0, 0])
#         #     else:
#         #         self.print_to_log_file('stem weight',self.network.encoder.stages[0][0].convs[0].conv.weight[0, 0, 0])
#         #         self.print_to_log_file('first fixed', self.network.encoder.stages[0][0].convs[1].conv.weight[0, 0, 0])
#
#
#         self.logger.log('epoch_start_timestamps', time(), self.current_epoch)
#
#     def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
#         if not self.was_initialized:
#             self.initialize()
#
#         if isinstance(filename_or_checkpoint, str):
#             checkpoint = torch.load(filename_or_checkpoint, map_location=self.device)
#         # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
#         # match. Use heuristic to make it match
#         new_state_dict = {}
#         for k, value in checkpoint['network_weights'].items():
#             key = k
#             if key not in self.network.state_dict().keys() and key.startswith('module.'):
#                 key = key[7:]
#             new_state_dict[key] = value
#
#         self.my_init_kwargs = checkpoint['init_args']
#         self.current_epoch = checkpoint['current_epoch']
#         self.logger.load_checkpoint(checkpoint['logging'])
#         self._best_ema = checkpoint['_best_ema']
#         self.inference_allowed_mirroring_axes = checkpoint[
#             'inference_allowed_mirroring_axes'] if 'inference_allowed_mirroring_axes' in checkpoint.keys() else self.inference_allowed_mirroring_axes
#
#         # messing with state dict naming schemes. Facepalm.
#         if self.is_ddp:
#             if isinstance(self.network.module, OptimizedModule):
#                 self.network.module._orig_mod.load_state_dict(new_state_dict)
#             else:
#                 self.network.module.load_state_dict(new_state_dict)
#         else:
#             if isinstance(self.network, OptimizedModule):
#                 self.network._orig_mod.load_state_dict(new_state_dict)
#             else:
#                 self.network.load_state_dict(new_state_dict)
#
#         if self.warmup_duration_heads + self.warmup_duration_whole_net >= self.current_epoch >= self.warmup_duration_heads:
#             self.optimizer, self.lr_scheduler = self.configure_optimizers(False, True)
#         if self.current_epoch >= self.warmup_duration_heads + self.warmup_duration_whole_net:
#             self.optimizer, self.lr_scheduler = self.configure_optimizers(False, False)
#
#         self.optimizer.load_state_dict(checkpoint['optimizer_state'])
#
#
#
#         if self.grad_scaler is not None:
#             if checkpoint['grad_scaler_state'] is not None:
#                 self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])
#
#
#     def _build_loss(self):
#         # set smooth to 0
#         if self.label_manager.has_regions:
#             loss = DC_and_BCE_loss({},
#                                    {'batch_dice': self.configuration_manager.batch_dice,
#                                     'do_bg': True, 'smooth': 0, 'ddp': self.is_ddp},
#                                    use_ignore_label=self.label_manager.ignore_label is not None,
#                                    dice_class=MemoryEfficientSoftDiceLoss)
#         else:
#             loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
#                                    'smooth': 0, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
#                                   ignore_label=self.label_manager.ignore_label,
#                                   dice_class=MemoryEfficientSoftDiceLoss)
#
#         deep_supervision_scales = self._get_deep_supervision_scales()
#
#         # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
#         # this gives higher resolution outputs more weight in the loss
#         weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
#
#         # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
#         weights = weights / weights.sum()
#         # now wrap the loss
#         loss = DeepSupervisionWrapper(loss, weights)
#         return loss
#
# class nnUNetTrainer_multipleinputs_warmupsegheads_nosmooth(nnUNetTrainerDiceCELoss_noSmooth):
#     def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
#                  device: torch.device = torch.device('cuda')):
#         # From https://grugbrain.dev/. Worth a read ya big brains ;-)
#
#         # apex predator of grug is complexity
#         # complexity bad
#         # say again:
#         # complexity very bad
#         # you say now:
#         # complexity very, very bad
#         # given choice between complexity or one on one against t-rex, grug take t-rex: at least grug see t-rex
#         # complexity is spirit demon that enter codebase through well-meaning but ultimately very clubbable non grug-brain developers and project managers who not fear complexity spirit demon or even know about sometime
#         # one day code base understandable and grug can get work done, everything good!
#         # next day impossible: complexity demon spirit has entered code and very dangerous situation!
#
#         # OK OK I am guilty. But I tried.
#         # https://www.osnews.com/images/comics/wtfm.jpg
#         # https://i.pinimg.com/originals/26/b2/50/26b250a738ea4abc7a5af4d42ad93af0.jpg
#
#         self.is_ddp = dist.is_available() and dist.is_initialized()
#         self.local_rank = 0 if not self.is_ddp else dist.get_rank()
#
#         self.device = device
#
#         # print what device we are using
#         if self.is_ddp:  # implicitly it's clear that we use cuda in this case
#             print(f"I am local rank {self.local_rank}. {device_count()} GPUs are available. The world size is "
#                   f"{dist.get_world_size()}."
#                   f"Setting device to {self.device}")
#             self.device = torch.device(type='cuda', index=self.local_rank)
#         else:
#             if self.device.type == 'cuda':
#                 # we might want to let the user pick this but for now please pick the correct GPU with CUDA_VISIBLE_DEVICES=X
#                 self.device = torch.device(type='cuda', index=0)
#             print(f"Using device: {self.device}")
#
#         # loading and saving this class for continuing from checkpoint should not happen based on pickling. This
#         # would also pickle the network etc. Bad, bad. Instead we just reinstantiate and then load the checkpoint we
#         # need. So let's save the init args
#         self.my_init_kwargs = {}
#         for k in inspect.signature(self.__init__).parameters.keys():
#             self.my_init_kwargs[k] = locals()[k]
#
#         ###  Saving all the init args into class variables for later access
#         self.plans_manager = PlansManager(plans)
#         self.configuration_manager = self.plans_manager.get_configuration(configuration)
#         self.configuration_name = configuration
#         self.dataset_json = dataset_json
#         self.fold = fold
#         self.unpack_dataset = unpack_dataset
#
#         ### Setting all the folder names. We need to make sure things don't crash in case we are just running
#         # inference and some of the folders may not be defined!
#         self.preprocessed_dataset_folder_base = join(nnUNet_preprocessed, self.plans_manager.dataset_name) \
#             if nnUNet_preprocessed is not None else None
#         self.output_folder_base = join(nnUNet_results, self.plans_manager.dataset_name,
#                                        self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" + configuration) \
#             if nnUNet_results is not None else None
#         if 'pretrain_name' in plans.keys():
#             pretrain_name = plans['pretrain_name']
#         else:
#             pretrain_name= None
#         if pretrain_name is not None:
#             self.output_folder_base = join(nnUNet_results, self.plans_manager.dataset_name, 'pretrained', pretrain_name,
#                                            self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" + configuration) \
#                 if nnUNet_results is not None else None
#             self.output_folder = join(self.output_folder_base, f'fold_{fold}')
#
#         else:
#             raise RuntimeError('No pretraining weights provided. Makes no sense to use this trainier')
#
#         self.preprocessed_dataset_folder = join(self.preprocessed_dataset_folder_base,
#                                                 self.configuration_manager.data_identifier)
#         # unlike the previous nnunet folder_with_segs_from_previous_stage is now part of the plans. For now it has to
#         # be a different configuration in the same plans
#         # IMPORTANT! the mapping must be bijective, so lowres must point to fullres and vice versa (using
#         # "previous_stage" and "next_stage"). Otherwise it won't work!
#         self.is_cascaded = self.configuration_manager.previous_stage_name is not None
#         self.folder_with_segs_from_previous_stage = \
#             join(nnUNet_results, self.plans_manager.dataset_name,
#                  self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" +
#                  self.configuration_manager.previous_stage_name, 'predicted_next_stage', self.configuration_name) \
#                 if self.is_cascaded else None
#
#         ### Some hyperparameters for you to fiddle with
#         self.initial_lr = 1e-2
#         self.weight_decay = 3e-5
#         self.oversample_foreground_percent = 0.33
#         self.num_iterations_per_epoch = 250
#         self.num_val_iterations_per_epoch = 50
#         self.num_epochs = 1000
#         self.current_epoch = 0
#         self.enable_deep_supervision = True
#
#         #### hyperparameter for warmup
#         self.warmup_duration_whole_net = 50  # lin increase whole network
#         self.warmup_heads_max_lr = 1e-3  # lin increase lr for heads
#         self.warmup_duration_heads = 10  # this is for the seg heads
#         self.num_epochs = 1000 + self.warmup_duration_whole_net + self.warmup_duration_heads
#         self.has_stem = False if self.configuration_manager.network_arch_class_name=="PlainConvUNet" else True
#         ### Dealing with labels/regions
#         self.label_manager = self.plans_manager.get_label_manager(dataset_json)
#         # labels can either be a list of int (regular training) or a list of tuples of int (region-based training)
#         # needed for predictions. We do sigmoid in case of (overlapping) regions
#
#         self.num_input_channels = None  # -> self.initialize()
#         self.network = None  # -> self.build_network_architecture()
#         self.optimizer = self.lr_scheduler = None  # -> self.initialize
#         self.grad_scaler = GradScaler() if self.device.type == 'cuda' else None
#         self.loss = None  # -> self.initialize
#
#         ### Simple logging. Don't take that away from me!
#         # initialize log file. This is just our log for the print statements etc. Not to be confused with lightning
#         # logging
#         timestamp = datetime.now()
#         maybe_mkdir_p(self.output_folder)
#         self.log_file = join(self.output_folder, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
#                              (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
#                               timestamp.second))
#         self.logger = nnUNetLogger()
#
#         ### placeholders
#         self.dataloader_train = self.dataloader_val = None  # see on_train_start
#
#         ### initializing stuff for remembering things and such
#         self._best_ema = None
#
#         ### inference things
#         self.inference_allowed_mirroring_axes = None  # this variable is set in
#         # self.configure_rotation_dummyDA_mirroring_and_inital_patch_size and will be saved in checkpoints
#
#         ### checkpoint saving stuff
#         self.save_every = 50
#         self.disable_checkpointing = False
#
#         ## DDP batch size and oversampling can differ between workers and needs adaptation
#         # we need to change the batch size in DDP because we don't use any of those distributed samplers
#         self._set_batch_size_and_oversample()
#
#         self.was_initialized = False
#
#         self.print_to_log_file("\n#######################################################################\n"
#                                "Please cite the following paper when using nnU-Net:\n"
#                                "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
#                                "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
#                                "Nature methods, 18(2), 203-211.\n"
#                                "#######################################################################\n",
#                                also_print_to_console=True, add_timestamp=False)
#
#
#     def initialize(self):
#         if not self.was_initialized:
#             self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
#                                                                    self.dataset_json)
#
#             self.network = self.build_network_architecture(
#                 self.configuration_manager.network_arch_class_name,
#                 self.configuration_manager.network_arch_init_kwargs,
#                 self.configuration_manager.network_arch_init_kwargs_req_import,
#                 self.num_input_channels,
#                 self.label_manager.num_segmentation_heads,
#                 self.enable_deep_supervision
#             ).to(self.device)
#             # compile network for free speedup
#             if self._do_i_compile():
#                 self.print_to_log_file('Using torch.compile...')
#                 self.network = torch.compile(self.network)
#
#             self.optimizer, self.lr_scheduler = self.configure_optimizers(True)
#             # if ddp, wrap in DDP wrapper
#             if self.is_ddp:
#                 self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
#                 self.network = DDP(self.network, device_ids=[self.local_rank])
#
#             self.loss = self._build_loss()
#             # torch 2.2.2 crashes upon compiling CE loss
#             # if self._do_i_compile():
#             #     self.loss = torch.compile(self.loss)
#             self.was_initialized = True
#         else:
#             raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
#                                "That should not happen.")
#
#     def configure_optimizers(self, seg_heads_only=False, netwarmup=False):
#
#         if isinstance(self.network, DDP):
#             params = self.network.module.parameters()
#             heads = self.network.module.decoder.seg_layers.parameters()
#             stem = self.network.module.encoder.stem.parameters()
#         else:
#             params = self.network.parameters()
#             # print(self.network.state_dict().keys())
#             heads = self.network.decoder.seg_layers.parameters()
#             if self.has_stem:
#                 stem = self.network.encoder.stem.parameters()
#             else:
#                 stem = self.network.encoder.stages[0][0].convs[0].parameters()
#         if seg_heads_only:
#             self.print_to_log_file("train only stem and heads")
#             optimizer = torch.optim.SGD(list(stem) +list(heads), self.initial_lr, weight_decay=self.weight_decay,
#                                         momentum=0.99, nesterov=True)
#             lr_scheduler = Lin_incr_LRScheduler(optimizer, self.warmup_heads_max_lr, self.warmup_duration_heads)
#         elif netwarmup and not seg_heads_only:
#             self.print_to_log_file("train whole net, warmup")
#             optimizer = torch.optim.SGD(params, self.initial_lr, weight_decay=self.weight_decay,
#                                         momentum=0.99, nesterov=True)
#             lr_scheduler = Lin_incr_offset_LRScheduler(optimizer, self.initial_lr, self.warmup_duration_whole_net, self.warmup_duration_heads)
#         else:
#             self.print_to_log_file("train whole net, default schedule")
#             optimizer = torch.optim.SGD(params, self.initial_lr, weight_decay=self.weight_decay,
#                                         momentum=0.99, nesterov=True)
#             lr_scheduler = PolyLRScheduler_offset(optimizer, self.initial_lr, self.num_epochs, self.warmup_duration_whole_net + self.warmup_duration_heads)
#         return optimizer, lr_scheduler
#
#     def on_train_epoch_start(self):
#         self.network.train()
#         if self.current_epoch == self.warmup_duration_heads:
#             self.optimizer, self.lr_scheduler = self.configure_optimizers(False, True)
#         if self.current_epoch == self.warmup_duration_heads + self.warmup_duration_whole_net:
#             self.optimizer, self.lr_scheduler = self.configure_optimizers(False, False)
#         self.lr_scheduler.step(self.current_epoch)
#         self.print_to_log_file('')
#         self.print_to_log_file(f'Epoch {self.current_epoch}')
#         self.print_to_log_file(
#             f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}")
#         # lrs are the same for all workers so we don't need to gather them in case of DDP training
#         self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)
#
#     def on_epoch_start(self):
#         # if isinstance(self.network, DDP):
#         #     self.print_to_log_file('seg output weight', self.network.module.decoder.seg_layers[0].weight[0,0])
#         #     if self.has_stem:
#         #         self.print_to_log_file('stem weight', self.network.module.encoder.stem.convs[0].conv.weight[0, 0])
#         #         self.print_to_log_file('first fixed weight', self.network.module.encoder.stages[0].blocks[0].conv1.conv.weight[0, 0, 0])
#         #     else:
#         #         self.print_to_log_file('stem weight',self.network.module.encoder.stages[0][0].convs[0].conv.weight[0, 0, 0])
#         #         self.print_to_log_file('first fixed weight', self.network.module.encoder.stages[0][0].convs[1].conv.weight[0, 0, 0])
#         # else:
#         #     self.print_to_log_file('seg output weight', self.network.decoder.seg_layers[0].weight[0,0])
#         #     if self.has_stem:
#         #         self.print_to_log_file('stem weight', self.network.encoder.stem.convs[0].conv.weight[0, 0])
#         #         self.print_to_log_file('free weight', self.network.encoder.stages[0].blocks[0].conv1.conv.weight[0, 0, 0])
#         #     else:
#         #         self.print_to_log_file('stem weight',self.network.encoder.stages[0][0].convs[0].conv.weight[0, 0, 0])
#         #         self.print_to_log_file('first fixed', self.network.encoder.stages[0][0].convs[1].conv.weight[0, 0, 0])
#
#
#         self.logger.log('epoch_start_timestamps', time(), self.current_epoch)
#
#     def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
#         if not self.was_initialized:
#             self.initialize()
#
#         if isinstance(filename_or_checkpoint, str):
#             checkpoint = torch.load(filename_or_checkpoint, map_location=self.device)
#         # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
#         # match. Use heuristic to make it match
#         new_state_dict = {}
#         for k, value in checkpoint['network_weights'].items():
#             key = k
#             if key not in self.network.state_dict().keys() and key.startswith('module.'):
#                 key = key[7:]
#             new_state_dict[key] = value
#
#         self.my_init_kwargs = checkpoint['init_args']
#         self.current_epoch = checkpoint['current_epoch']
#         self.logger.load_checkpoint(checkpoint['logging'])
#         self._best_ema = checkpoint['_best_ema']
#         self.inference_allowed_mirroring_axes = checkpoint[
#             'inference_allowed_mirroring_axes'] if 'inference_allowed_mirroring_axes' in checkpoint.keys() else self.inference_allowed_mirroring_axes
#
#         # messing with state dict naming schemes. Facepalm.
#         if self.is_ddp:
#             if isinstance(self.network.module, OptimizedModule):
#                 self.network.module._orig_mod.load_state_dict(new_state_dict)
#             else:
#                 self.network.module.load_state_dict(new_state_dict)
#         else:
#             if isinstance(self.network, OptimizedModule):
#                 self.network._orig_mod.load_state_dict(new_state_dict)
#             else:
#                 self.network.load_state_dict(new_state_dict)
#
#         if self.warmup_duration_heads + self.warmup_duration_whole_net >= self.current_epoch >= self.warmup_duration_heads:
#             self.optimizer, self.lr_scheduler = self.configure_optimizers(False, True)
#         if self.current_epoch >= self.warmup_duration_heads + self.warmup_duration_whole_net:
#             self.optimizer, self.lr_scheduler = self.configure_optimizers(False, False)
#
#         self.optimizer.load_state_dict(checkpoint['optimizer_state'])
#
#
#
#         if self.grad_scaler is not None:
#             if checkpoint['grad_scaler_state'] is not None:
#                 self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])
#
#
#     def _build_loss(self):
#         # set smooth to 0
#         if self.label_manager.has_regions:
#             loss = DC_and_BCE_loss({},
#                                    {'batch_dice': self.configuration_manager.batch_dice,
#                                     'do_bg': True, 'smooth': 0, 'ddp': self.is_ddp},
#                                    use_ignore_label=self.label_manager.ignore_label is not None,
#                                    dice_class=MemoryEfficientSoftDiceLoss)
#         else:
#             loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
#                                    'smooth': 0, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
#                                   ignore_label=self.label_manager.ignore_label,
#                                   dice_class=MemoryEfficientSoftDiceLoss)
#
#         deep_supervision_scales = self._get_deep_supervision_scales()
#
#         # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
#         # this gives higher resolution outputs more weight in the loss
#         weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
#
#         # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
#         weights = weights / weights.sum()
#         # now wrap the loss
#         loss = DeepSupervisionWrapper(loss, weights)
#         return loss
#
# class nnUNetTrainer_multipleinputs_warmupdecoder_nosmooth(nnUNetTrainer_warmupdecoder_nosmooth):
#     def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
#                  device: torch.device = torch.device('cuda')):
#         # From https://grugbrain.dev/. Worth a read ya big brains ;-)
#
#         # apex predator of grug is complexity
#         # complexity bad
#         # say again:
#         # complexity very bad
#         # you say now:
#         # complexity very, very bad
#         # given choice between complexity or one on one against t-rex, grug take t-rex: at least grug see t-rex
#         # complexity is spirit demon that enter codebase through well-meaning but ultimately very clubbable non grug-brain developers and project managers who not fear complexity spirit demon or even know about sometime
#         # one day code base understandable and grug can get work done, everything good!
#         # next day impossible: complexity demon spirit has entered code and very dangerous situation!
#
#         # OK OK I am guilty. But I tried.
#         # https://www.osnews.com/images/comics/wtfm.jpg
#         # https://i.pinimg.com/originals/26/b2/50/26b250a738ea4abc7a5af4d42ad93af0.jpg
#
#         self.is_ddp = dist.is_available() and dist.is_initialized()
#         self.local_rank = 0 if not self.is_ddp else dist.get_rank()
#
#         self.device = device
#
#         # print what device we are using
#         if self.is_ddp:  # implicitly it's clear that we use cuda in this case
#             print(f"I am local rank {self.local_rank}. {device_count()} GPUs are available. The world size is "
#                   f"{dist.get_world_size()}."
#                   f"Setting device to {self.device}")
#             self.device = torch.device(type='cuda', index=self.local_rank)
#         else:
#             if self.device.type == 'cuda':
#                 # we might want to let the user pick this but for now please pick the correct GPU with CUDA_VISIBLE_DEVICES=X
#                 self.device = torch.device(type='cuda', index=0)
#             print(f"Using device: {self.device}")
#
#         # loading and saving this class for continuing from checkpoint should not happen based on pickling. This
#         # would also pickle the network etc. Bad, bad. Instead we just reinstantiate and then load the checkpoint we
#         # need. So let's save the init args
#         self.my_init_kwargs = {}
#         for k in inspect.signature(self.__init__).parameters.keys():
#             self.my_init_kwargs[k] = locals()[k]
#
#         ###  Saving all the init args into class variables for later access
#         self.plans_manager = PlansManager(plans)
#         self.configuration_manager = self.plans_manager.get_configuration(configuration)
#         self.configuration_name = configuration
#         self.dataset_json = dataset_json
#         self.fold = fold
#         self.unpack_dataset = unpack_dataset
#
#         ### Setting all the folder names. We need to make sure things don't crash in case we are just running
#         # inference and some of the folders may not be defined!
#         self.preprocessed_dataset_folder_base = join(nnUNet_preprocessed, self.plans_manager.dataset_name) \
#             if nnUNet_preprocessed is not None else None
#         self.output_folder_base = join(nnUNet_results, self.plans_manager.dataset_name,
#                                        self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" + configuration) \
#             if nnUNet_results is not None else None
#         if 'pretrain_name' in plans.keys():
#             pretrain_name = plans['pretrain_name']
#         else:
#             pretrain_name= None
#         if pretrain_name is not None:
#             self.output_folder_base = join(nnUNet_results, self.plans_manager.dataset_name, 'pretrained', pretrain_name,
#                                            self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" + configuration) \
#                 if nnUNet_results is not None else None
#             self.output_folder = join(self.output_folder_base, f'fold_{fold}')
#
#         else:
#             raise RuntimeError('No pretraining weights provided. Makes no sense to use this trainier')
#
#         self.preprocessed_dataset_folder = join(self.preprocessed_dataset_folder_base,
#                                                 self.configuration_manager.data_identifier)
#         # unlike the previous nnunet folder_with_segs_from_previous_stage is now part of the plans. For now it has to
#         # be a different configuration in the same plans
#         # IMPORTANT! the mapping must be bijective, so lowres must point to fullres and vice versa (using
#         # "previous_stage" and "next_stage"). Otherwise it won't work!
#         self.is_cascaded = self.configuration_manager.previous_stage_name is not None
#         self.folder_with_segs_from_previous_stage = \
#             join(nnUNet_results, self.plans_manager.dataset_name,
#                  self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" +
#                  self.configuration_manager.previous_stage_name, 'predicted_next_stage', self.configuration_name) \
#                 if self.is_cascaded else None
#
#         ### Some hyperparameters for you to fiddle with
#         self.initial_lr = 1e-2
#         self.weight_decay = 3e-5
#         self.oversample_foreground_percent = 0.33
#         self.num_iterations_per_epoch = 250
#         self.num_val_iterations_per_epoch = 50
#         self.num_epochs = 1000
#         self.current_epoch = 0
#         self.enable_deep_supervision = True
#
#         #### hyperparameter for warmup
#         self.warmup_duration_whole_net = 50  # lin increase whole network
#         self.warmup_heads_max_lr = 1e-3  # lin increase lr for heads
#         self.warmup_duration_heads = 50  # this is for the DECODER
#         self.num_epochs = 1000 + self.warmup_duration_whole_net + self.warmup_duration_heads
#         self.has_stem = False if self.configuration_manager.network_arch_class_name=="PlainConvUNet" else True
#
#         ### Dealing with labels/regions
#         self.label_manager = self.plans_manager.get_label_manager(dataset_json)
#         # labels can either be a list of int (regular training) or a list of tuples of int (region-based training)
#         # needed for predictions. We do sigmoid in case of (overlapping) regions
#
#         self.num_input_channels = None  # -> self.initialize()
#         self.network = None  # -> self.build_network_architecture()
#         self.optimizer = self.lr_scheduler = None  # -> self.initialize
#         self.grad_scaler = GradScaler() if self.device.type == 'cuda' else None
#         self.loss = None  # -> self.initialize
#
#         ### Simple logging. Don't take that away from me!
#         # initialize log file. This is just our log for the print statements etc. Not to be confused with lightning
#         # logging
#         timestamp = datetime.now()
#         maybe_mkdir_p(self.output_folder)
#         self.log_file = join(self.output_folder, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
#                              (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
#                               timestamp.second))
#         self.logger = nnUNetLogger()
#
#         ### placeholders
#         self.dataloader_train = self.dataloader_val = None  # see on_train_start
#
#         ### initializing stuff for remembering things and such
#         self._best_ema = None
#
#         ### inference things
#         self.inference_allowed_mirroring_axes = None  # this variable is set in
#         # self.configure_rotation_dummyDA_mirroring_and_inital_patch_size and will be saved in checkpoints
#
#         ### checkpoint saving stuff
#         self.save_every = 50
#         self.disable_checkpointing = False
#
#         ## DDP batch size and oversampling can differ between workers and needs adaptation
#         # we need to change the batch size in DDP because we don't use any of those distributed samplers
#         self._set_batch_size_and_oversample()
#
#         self.was_initialized = False
#
#         self.print_to_log_file("\n#######################################################################\n"
#                                "Please cite the following paper when using nnU-Net:\n"
#                                "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
#                                "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
#                                "Nature methods, 18(2), 203-211.\n"
#                                "#######################################################################\n",
#                                also_print_to_console=True, add_timestamp=False)
#
#     def configure_optimizers(self, dec_only=False, netwarmup=False):
#
#         if isinstance(self.network, DDP):
#             params = self.network.module.parameters()
#             dec = self.network.module.decoder.parameters()
#             stem = self.network.module.encoder.stem.parameters()
#         else:
#             params = self.network.parameters()
#             # print(self.network.state_dict().keys())
#             dec = self.network.decoder.parameters()
#             if self.has_stem:
#                 stem = self.network.encoder.stem.parameters()
#             else:
#                 stem = self.network.encoder.stages[0][0].convs[0].parameters()
#         if dec_only:
#             self.print_to_log_file("train only decoder")
#             optimizer = torch.optim.SGD(list(stem) +list(dec), self.initial_lr, weight_decay=self.weight_decay,
#                                         momentum=0.99, nesterov=True)
#             lr_scheduler = Lin_incr_LRScheduler(optimizer, self.warmup_heads_max_lr, self.warmup_duration_heads)
#         elif netwarmup and not dec_only:
#             self.print_to_log_file("train whole net, warmup")
#             optimizer = torch.optim.SGD(params, self.initial_lr, weight_decay=self.weight_decay,
#                                         momentum=0.99, nesterov=True)
#             lr_scheduler = Lin_incr_offset_LRScheduler(optimizer, self.initial_lr, self.warmup_duration_whole_net, self.warmup_duration_heads)
#         else:
#             self.print_to_log_file("train whole net, default schedule")
#             optimizer = torch.optim.SGD(params, self.initial_lr, weight_decay=self.weight_decay,
#                                         momentum=0.99, nesterov=True)
#             lr_scheduler = PolyLRScheduler_offset(optimizer, self.initial_lr, self.num_epochs, self.warmup_duration_whole_net + self.warmup_duration_heads)
#         return optimizer, lr_scheduler
#     def on_epoch_start(self):
#         if isinstance(self.network, DDP):
#             self.print_to_log_file('seg output weight', self.network.module.decoder.seg_layers[0].weight[0,0])
#             if self.has_stem:
#                 self.print_to_log_file('stem weight', self.network.module.encoder.stem.convs[0].conv.weight[0, 0])
#                 self.print_to_log_file('first fixed weight', self.network.module.encoder.stages[0].blocks[0].conv1.conv.weight[0, 0, 0])
#             else:
#                 self.print_to_log_file('stem weight',self.network.module.encoder.stages[0][0].convs[0].conv.weight[0, 0, 0])
#                 self.print_to_log_file('first fixed weight', self.network.module.encoder.stages[0][0].convs[1].conv.weight[0, 0, 0])
#         else:
#             self.print_to_log_file('seg output weight', self.network.decoder.seg_layers[0].weight[0,0])
#             if self.has_stem:
#                 self.print_to_log_file('stem weight', self.network.encoder.stem.convs[0].conv.weight[0, 0])
#                 self.print_to_log_file('first fixed weight', self.network.encoder.stages[0].blocks[0].conv1.conv.weight[0, 0, 0])
#             else:
#                 self.print_to_log_file('stem weight',self.network.encoder.stages[0][0].convs[0].conv.weight[0, 0, 0])
#                 self.print_to_log_file('first fixed weight', self.network.encoder.stages[0][0].convs[1].conv.weight[0, 0, 0])
#
#
#         self.logger.log('epoch_start_timestamps', time(), self.current_epoch)
#
# class nnUNetTrainer_warmupnet(nnUNetTrainer_pretrained):
#     def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset =True,
#                  device: torch.device = torch.device('cuda')):
#         super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
#         #### hyperparameters for warmup
#         self.warmup_duration_whole_net = 50  # lin increase whole network
#         self.num_epochs = 1000 + self.warmup_duration_whole_net
#         self.training_stage = None  # 'warmup_all', 'train'
#
#     def configure_optimizers(self, stage: str = 'warmup_all'):
#         assert stage in ['warmup_all', 'train']
#
#         if self.training_stage == stage:
#             return self.optimizer, self.lr_scheduler
#
#         if isinstance(self.network, DDP):
#             params = self.network.module.parameters()
#         else:
#             params = self.network.parameters()
#
#         if stage == 'warmup_all':
#             self.print_to_log_file("train whole net, warmup")
#             optimizer = torch.optim.SGD(params, self.initial_lr, weight_decay=self.weight_decay,
#                                         momentum=0.99, nesterov=True)
#             lr_scheduler = Lin_incr_LRScheduler(optimizer, self.initial_lr, self.warmup_duration_whole_net)
#             self.print_to_log_file(f"Initialized warmup_all optimizer and lr_scheduler at epoch {self.current_epoch}")
#         else:
#             self.print_to_log_file("train whole net, default schedule")
#             if self.training_stage == 'warmup_all':
#                 # we can keep the existing optimizer and don't need to create a new one. This will allow us to keep
#                 # the accumulated momentum terms which already point in a useful driection
#                 optimizer = self.optimizer
#             else:
#                 optimizer = torch.optim.SGD(params, self.initial_lr, weight_decay=self.weight_decay,
#                                             momentum=0.99, nesterov=True)
#             lr_scheduler = PolyLRScheduler_offset(optimizer, self.initial_lr, self.num_epochs, self.warmup_duration_whole_net)
#             self.print_to_log_file(f"Initialized train optimizer and lr_scheduler at epoch {self.current_epoch}")
#         self.training_stage = stage
#         empty_cache(self.device)
#         return optimizer, lr_scheduler
#
#     def on_train_epoch_start(self):
#         if self.current_epoch == 0:
#             self.optimizer, self.lr_scheduler = self.configure_optimizers('warmup_all')
#         elif self.current_epoch == self.warmup_duration_whole_net:
#             self.optimizer, self.lr_scheduler = self.configure_optimizers('train')
#
#         super().on_train_epoch_start()
#
#     def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
#         """
#         We need to overwrite that entire function because we need to fiddle the correct optimizer in between
#         loading the checkpoint and applying the optimizer states. Yuck.
#         """
#         if not self.was_initialized:
#             self.initialize()
#
#         if isinstance(filename_or_checkpoint, str):
#             checkpoint = torch.load(filename_or_checkpoint, map_location=self.device)
#         # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
#         # match. Use heuristic to make it match
#         new_state_dict = {}
#         for k, value in checkpoint['network_weights'].items():
#             key = k
#             if key not in self.network.state_dict().keys() and key.startswith('module.'):
#                 key = key[7:]
#             new_state_dict[key] = value
#
#         self.my_init_kwargs = checkpoint['init_args']
#         self.current_epoch = checkpoint['current_epoch']
#         self.logger.load_checkpoint(checkpoint['logging'])
#         self._best_ema = checkpoint['_best_ema']
#         self.inference_allowed_mirroring_axes = checkpoint[
#             'inference_allowed_mirroring_axes'] if 'inference_allowed_mirroring_axes' in checkpoint.keys() \
#             else self.inference_allowed_mirroring_axes
#
#         # messing with state dict naming schemes. Facepalm.
#         if self.is_ddp:
#             if isinstance(self.network.module, OptimizedModule):
#                 self.network.module._orig_mod.load_state_dict(new_state_dict)
#             else:
#                 self.network.module.load_state_dict(new_state_dict)
#         else:
#             if isinstance(self.network, OptimizedModule):
#                 self.network._orig_mod.load_state_dict(new_state_dict)
#             else:
#                 self.network.load_state_dict(new_state_dict)
#
#         # it's fine to do this every time we load because configure_optimizers will be a no-op if the correct optimizer
#         # and lr scheduler are already set up
#         if self.current_epoch < self.warmup_duration_whole_net:
#             self.optimizer, self.lr_scheduler = self.configure_optimizers('warmup_all')
#         else:
#             self.optimizer, self.lr_scheduler = self.configure_optimizers('train')
#
#         self.optimizer.load_state_dict(checkpoint['optimizer_state'])
#         if self.grad_scaler is not None:
#             if checkpoint['grad_scaler_state'] is not None:
#                 self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])
#
# class nnUNetTrainer_warmupnet_nosmooth(nnUNetTrainer_warmupnet):
#     def _build_loss(self):
#         # set smooth to 0
#         if self.label_manager.has_regions:
#             loss = DC_and_BCE_loss({},
#                                    {'batch_dice': self.configuration_manager.batch_dice,
#                                     'do_bg': True, 'smooth': 0, 'ddp': self.is_ddp},
#                                    use_ignore_label=self.label_manager.ignore_label is not None,
#                                    dice_class=MemoryEfficientSoftDiceLoss)
#         else:
#             loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
#                                    'smooth': 0, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
#                                   ignore_label=self.label_manager.ignore_label,
#                                   dice_class=MemoryEfficientSoftDiceLoss)
#
#         if self.enable_deep_supervision:
#             deep_supervision_scales = self._get_deep_supervision_scales()
#
#             # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
#             # this gives higher resolution outputs more weight in the loss
#             weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
#             weights[-1] = 0
#
#             # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
#             weights = weights / weights.sum()
#             # now wrap the loss
#             loss = DeepSupervisionWrapper(loss, weights)
#         return loss
#
#
# class nnUNetTrainer_warmupnet_1e3(nnUNetTrainer_warmupnet):
#     def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
#                  device: torch.device = torch.device('cuda')):
#         super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
#         self.initial_lr = 1e-3
#
# class nnUNetTrainer_warmupnet_1e3_nosmooth(nnUNetTrainer_warmupnet_nosmooth):
#     def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
#                  device: torch.device = torch.device('cuda')):
#         super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
#         self.initial_lr = 1e-3
#
# class nnUNetTrainer_warmupnet_dublicate(nnUNetTrainer_warmupnet):
#     def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
#                  device: torch.device = torch.device('cuda')):
#         super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
#
#
# class nnUNetTrainer_warmupnet_1e3_dublicate(nnUNetTrainer_warmupnet_1e3):
#     def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
#                  device: torch.device = torch.device('cuda')):
#         super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
#         self.initial_lr = 1e-3