from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
from collections import OrderedDict
from multiprocessing import Pool
import numpy as np
import torch
from _warnings import warn
from tqdm import trange
from time import sleep, time
from torch.backends import cudnn
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.dataset_conversion.Task100_MultiTalent import MultiTalent_regions, MultiTalent_region_output_idx_mapping, MultiTalent_regions_class_order, MultiTalent_valid_regions
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.paths import preprocessing_output_dir
from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from torch import distributed
from torch import nn
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Tuple
from nnunet.configuration import default_num_threads
from nnunet.evaluation.evaluator import aggregate_scores
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax
from nnunet.training.dataloading.dataset_loading import unpack_dataset, DataLoader3D
from nnunet.training.network_training.nnUNetTrainerV2_DDP import nnUNetTrainerV2_DDP
from nnunet.utilities.distributed import awesome_allgather_function
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from nnunet.network_architecture.generic_modular_residual_UNet import FabiansUNet, get_default_network_config
from nnunet.network_architecture.custom_modules.conv_blocks import BasicResidualBlock

def init_last_bn_before_add_to_0(module):
    if isinstance(module, BasicResidualBlock):
        module.norm2.weight = nn.init.constant_(module.norm2.weight, 0)
        module.norm2.bias = nn.init.constant_(module.norm2.bias, 0)

class MultiTalent_trainer_resenc_ddp(
    nnUNetTrainerV2_DDP
):
    def __init__(
        self,
        plans_file,
        fold,
        local_rank,
        output_folder=None,
        dataset_directory=None,
        batch_dice=True,
        stage=None,
        unpack_data=True,
        deterministic=True,
        distribute_batch_size=False,
        fp16=False,
    ):
        batch_dice = True
        super().__init__(
            plans_file=plans_file,
            fold=fold,
            local_rank=local_rank,
            output_folder=output_folder,
            dataset_directory=dataset_directory,
            batch_dice=batch_dice,
            stage=stage,
            unpack_data=unpack_data,
            deterministic=deterministic,
            fp16=fp16,
            distribute_batch_size=distribute_batch_size,
        )
        self.regions = MultiTalent_regions
        self.loss = None  # DC_and_BCE_loss_regions2({}, batch_dice=batch_dice)
        self.ce_loss = nn.BCEWithLogitsLoss()


    def initialize_network(self):
        """inference_apply_nonlin to sigmoid"""
        if self.threeD:
            cfg = get_default_network_config(3, None, norm_type="in")

        else:
            cfg = get_default_network_config(1, None, norm_type="in")

        stage_plans = self.plans["plans_per_stage"][self.stage]
        conv_kernel_sizes = stage_plans["conv_kernel_sizes"]
        blocks_per_stage_encoder = stage_plans["num_blocks_encoder"]
        blocks_per_stage_decoder = stage_plans["num_blocks_decoder"]
        pool_op_kernel_sizes = stage_plans["pool_op_kernel_sizes"]

        self.network = FabiansUNet(
            self.num_input_channels,
            self.base_num_features,
            blocks_per_stage_encoder,
            2,
            pool_op_kernel_sizes,
            conv_kernel_sizes,
            cfg,
            self.num_classes,
            blocks_per_stage_decoder,
            True,
            False,
            320,
            InitWeights_He(1e-2),
        )
        self.network.apply(init_last_bn_before_add_to_0)
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = nn.Sigmoid()


    def setup_DA_params(self):
        """
        net_num_pool_op_kernel_sizes is different in resunet
        """
        super().setup_DA_params()
        self.deep_supervision_scales = [[1, 1, 1]] + list(
            list(i)
            for i in 1
            / np.cumprod(np.vstack(self.net_num_pool_op_kernel_sizes[1:]), axis=0)
        )[:-1]

    def process_plans(self, plans):
        super().process_plans(plans)
        self.num_classes = len(self.regions)

    def initialize(self, training=True, force_load_plans=False):
        """
        order_seg = 0 or the CPU dies
        :param training:
        :param force_load_plans:
        :return:
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)

            self.setup_DA_params()

            self.folder_with_preprocessed_data = join(
                self.dataset_directory,
                self.plans["data_identifier"] + "_stage%d" % self.stage,
            )
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
                        "will wait all winter for your model to finish!"
                    )

                # setting weights for deep supervision losses
                net_numpool = len(self.net_num_pool_op_kernel_sizes)

                # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
                # this gives higher resolution outputs more weight in the loss
                weights = np.array([1 / (2**i) for i in range(net_numpool)])

                # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
                mask = np.array(
                    [True if i < net_numpool - 1 else False for i in range(net_numpool)]
                )
                weights[~mask] = 0
                weights = weights / weights.sum()
                self.ds_loss_weights = weights

                seeds_train = np.random.random_integers(
                    0, 99999, self.data_aug_params.get("num_threads")
                )
                seeds_val = np.random.random_integers(
                    0, 99999, max(self.data_aug_params.get("num_threads") // 2, 1)
                )
                print("seeds train", seeds_train)
                print("seeds_val", seeds_val)
                self.tr_gen, self.val_gen = get_moreDA_augmentation(
                    self.dl_tr,
                    self.dl_val,
                    self.data_aug_params["patch_size_for_spatialtransform"],
                    self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales,
                    seeds_train=seeds_train,
                    seeds_val=seeds_val,
                    pin_memory=self.pin_memory,
                    order_seg=0,
                )
                self.print_to_log_file(
                    "TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                    also_print_to_console=False,
                )
                self.print_to_log_file(
                    "VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                    also_print_to_console=False,
                )
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()
            self.network = DDP(self.network, device_ids=[self.local_rank])

        else:
            self.print_to_log_file(
                "self.was_initialized is True, not running self.initialize again"
            )
        self.was_initialized = True

        self.regions_class_order = list(range(self.num_classes))

    def validate(
        self,
        do_mirroring: bool = True,
        use_sliding_window: bool = True,
        step_size: int = 0.5,
        save_softmax: bool = True,
        use_gaussian: bool = True,
        overwrite: bool = True,
        validation_folder_name: str = "validation_raw",
        debug: bool = False,
        all_in_gpu: bool = False,
        segmentation_export_kwargs: dict = None,
        run_postprocessing_on_folds: bool = False,
    ):
        """
        run_postprocessing_on_folds IS IGNORED!!!! There is no postprocessing here!
        """



        if isinstance(self.network, DDP):
            net = self.network.module
        else:
            net = self.network

        net.do_ds = net.decoder.deep_supervision
        ds = net.decoder.deep_supervision
        net.decoder.deep_supervision = False

        current_mode = self.network.training

        validation_folder_name_individual = validation_folder_name + '_individual'

        self.network.eval()

        assert self.was_initialized, "must initialize, ideally with checkpoint (or train first)"
        if self.dataset_val is None:
            self.load_dataset()
            self.do_split()

        if segmentation_export_kwargs is None:
            if 'segmentation_export_params' in self.plans.keys():
                force_separate_z = self.plans['segmentation_export_params']['force_separate_z']
                interpolation_order = self.plans['segmentation_export_params']['interpolation_order']
                interpolation_order_z = self.plans['segmentation_export_params']['interpolation_order_z']
            else:
                force_separate_z = None
                interpolation_order = 1
                interpolation_order_z = 0
        else:
            force_separate_z = segmentation_export_kwargs['force_separate_z']
            interpolation_order = segmentation_export_kwargs['interpolation_order']
            interpolation_order_z = segmentation_export_kwargs['interpolation_order_z']

        # predictions as they come from the network go here
        output_folder = join(self.output_folder, validation_folder_name)
        maybe_mkdir_p(output_folder)
        output_folder_individual = join(self.output_folder, validation_folder_name_individual)
        maybe_mkdir_p(output_folder_individual)
        # this is for debug purposes
        my_input_args = {'do_mirroring': do_mirroring,
                         'use_sliding_window': use_sliding_window,
                         'step_size': step_size,
                         'save_softmax': save_softmax,
                         'use_gaussian': use_gaussian,
                         'overwrite': overwrite,
                         'validation_folder_name': validation_folder_name,
                         'debug': debug,
                         'all_in_gpu': all_in_gpu,
                         'segmentation_export_kwargs': segmentation_export_kwargs,
                         }
        save_json(my_input_args, join(output_folder, "validation_args.json"))

        if do_mirroring:
            if not self.data_aug_params['do_mirror']:
                raise RuntimeError("We did not train with mirroring so you cannot do inference with mirroring enabled")
            mirror_axes = self.data_aug_params['mirror_axes']
        else:
            mirror_axes = ()

        pred_gt_tuples = {}
        valid_labels = {}

        pool = Pool(8)

        all_keys = list(self.dataset_val.keys())
        my_keys = all_keys[self.local_rank::distributed.get_world_size()]
        # we cannot simply iterate over all_keys because we need to know pred_gt_tuples and valid_labels of all cases
        # for evaluation (which is done by local rank 0)
        for k in all_keys:
            r = []
            properties = load_pickle(self.dataset[k]['properties_file'])

            dataset_name = [i for i in MultiTalent_valid_regions.keys() if
                            i.startswith("Task%03.0d_" % int(k.split('_')[0]))]
            assert len(dataset_name) == 1
            dataset_name = dataset_name[0]
            if dataset_name not in valid_labels:
                valid_labels[dataset_name] = properties['valid_labels']
            if dataset_name not in pred_gt_tuples.keys():
                pred_gt_tuples[dataset_name] = []

            fname = properties['list_of_data_files'][0].split("/")[-1][:-12]
            pred_gt_tuples[dataset_name].append([join(output_folder, fname + ".nii.gz"),
                                   join(self.gt_niftis_folder, fname + ".nii.gz")])

            # figure out if we need to predict the case
            if overwrite:
                prediction_needed = True
            else:
                prediction_needed = False
                if not isfile(join(output_folder, fname + ".nii.gz")):
                    prediction_needed = True
                if save_softmax and not isfile(join(output_folder, fname + ".npz")):
                    prediction_needed = True
                for l in MultiTalent_regions.keys():
                    region_name = l
                    fname_here = properties['list_of_data_files'][0].split("/")[-1][:-12] + '__' + region_name
                    output_fname_nii_gz = join(output_folder_individual, fname_here + ".nii.gz")
                    if not isfile(output_fname_nii_gz):
                        prediction_needed = True

            if prediction_needed and k in my_keys:
                data = np.load(self.dataset[k]['data_file'])['data']

                print(k, data.shape)
                data[-1][data[-1] == -1] = 0
                softmax_pred = self.predict_preprocessed_data_return_seg_and_softmax(data[:-1], do_mirroring,
                                                                                     mirror_axes, use_sliding_window,
                                                                                     step_size, use_gaussian,
                                                                                     all_in_gpu=all_in_gpu,
                                                                                     mixed_precision=self.fp16)[1]

                softmax_pred = softmax_pred.transpose([0] + [i + 1 for i in self.transpose_backward])

                if save_softmax:
                    softmax_fname = join(output_folder, fname + ".npz")
                else:
                    softmax_fname = None

                """There is a problem with python process communication that prevents us from communicating obejcts
                larger than 2 GB between processes (basically when the length of the pickle string that will be sent is
                communicated by the multiprocessing.Pipe object then the placeholder (\%i I think) does not allow for long
                enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually
                patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will
                then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either
                filename or np.ndarray and will handle this automatically"""
                # export individual labels

                for l in MultiTalent_regions.keys():
                    region_name = l
                    fname_here = properties['list_of_data_files'][0].split("/")[-1][:-12] + '__' + region_name

                    softmax_pred_here = softmax_pred[MultiTalent_region_output_idx_mapping[l]][None]
                    output_fname_nii_gz = join(output_folder_individual,fname_here + ".nii.gz")
                    r.append(pool.starmap_async(save_segmentation_nifti_from_softmax, ((softmax_pred_here, output_fname_nii_gz,
                                                                   properties, interpolation_order, ((1,),),
                                                                   None, None,
                                                                   None, None, force_separate_z,
                                                                   interpolation_order_z), )))

                # save_segmentation_nifti_from_softmax(softmax_pred_here, output_fname_nii_gz,
                #                                                properties, interpolation_order, ((1,),),
                #                                                None, None,
                #                                                None, None, force_separate_z,
                #                                                interpolation_order_z)

                # export all labels of this dataset
                valid_regions = [MultiTalent_region_output_idx_mapping[i] for i in MultiTalent_valid_regions[dataset_name]]
                softmax_pred = softmax_pred[valid_regions]

                regions_class_order = MultiTalent_regions_class_order[dataset_name]
                r.append(pool.starmap_async(save_segmentation_nifti_from_softmax, ((softmax_pred, join(output_folder, fname + ".nii.gz"),
                                                           properties, interpolation_order, regions_class_order,
                                                           None, None,
                                                           softmax_fname, None, force_separate_z,
                                                           interpolation_order_z), )))
                # save_segmentation_nifti_from_softmax(softmax_pred, join(output_folder, fname + ".nii.gz"),
                #                                            properties, interpolation_order, regions_class_order,
                #                                            None, None,
                #                                            softmax_fname, None, force_separate_z,
                #                                            interpolation_order_z)
                _ = [i.get() for i in r]
        pool.close()
        pool.join()

        distributed.barrier()

        self.print_to_log_file("finished prediction")

        # evaluate raw predictions
        if self.local_rank == 0:
            self.print_to_log_file("evaluation of raw predictions")
            task = self.dataset_directory.split("/")[-1]
            job_name = self.experiment_name
            for dataset in pred_gt_tuples.keys():
                _ = aggregate_scores(pred_gt_tuples[dataset], labels=valid_labels[dataset],
                                     json_output_file=join(output_folder, "summary_%s.json" % dataset),
                                     json_name=job_name + " val tiled %s" % (str(use_sliding_window)),
                                     json_author="Fabian",
                                     json_task=task, num_threads=max(64, default_num_threads * distributed.get_world_size()), )

        # automatic postprocessing is omitted because that would be a heck of a lot of work to get it right. We can
        # add it manually if we want
        # automatic postprocessing is omitted because that would be a heck of a lot of work to get it right. We can
        # add it manually if we want

        self.network.train(current_mode)
        net.decoder.deep_supervision = ds

        distributed.barrier()





    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True,
                                                         pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                         verbose: bool = True, mixed_precision=True,
                                                         region_vec=None) -> Tuple[
        np.ndarray, np.ndarray]:



        if pad_border_mode == 'constant' and pad_kwargs is None:
            pad_kwargs = {'constant_values': 0}

        if do_mirroring and mirror_axes is None:
            mirror_axes = self.data_aug_params['mirror_axes']

        if do_mirroring:
            assert self.data_aug_params["do_mirror"], "Cannot do mirroring as test time augmentation when training " \
                                                      "was done without mirroring"

        valid = list((SegmentationNetwork, nn.DataParallel, DDP))
        assert isinstance(self.network, tuple(valid))
        if isinstance(self.network, DDP):
            net = self.network.module
        else:
            net = self.network
        net.do_ds = net.decoder.deep_supervision
        ds = net.decoder.deep_supervision
        net.decoder.deep_supervision = False
        net.eval()
        ret = net.predict_3D(data, do_mirroring=do_mirroring, mirror_axes=mirror_axes,
                             use_sliding_window=use_sliding_window, step_size=step_size,
                             patch_size=self.patch_size, regions_class_order=self.regions_class_order,
                             use_gaussian=use_gaussian, pad_border_mode=pad_border_mode,
                             pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu, verbose=verbose,
                             mixed_precision=mixed_precision, region_vec=region_vec)
        net.decoder.deep_supervision = ds
        net.do_ds = ds


        return ret

    def run_iteration(
        self, data_generator, do_backprop=True, run_online_evaluation=False
    ):
        data_dict = next(data_generator)
        data = data_dict["data"]
        target = data_dict["target"]
        props = data_dict["properties"]
        valid_regions = [p["valid_regions"] for p in props]
        # print(np.unique(target[0]),[p['list_of_data_files'] for p in props])

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
                l, ce, dc = self.compute_loss(output, target, valid_regions)

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            del data
            l, ce, dc = self.compute_loss(output, target, valid_regions)

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target, valid_regions)

        del target

        return (
            l.detach().cpu().numpy(),
            ce.detach().cpu().numpy(),
            dc.detach().cpu().numpy(),
        )

    def run_online_evaluation(self, output, target, valid_regions):
        output = output[0]
        target = target[0]
        with torch.no_grad():
            out_sigmoid = torch.sigmoid(output)
            out_sigmoid = (out_sigmoid > 0.5).float()

            shp_output = output.shape

            tp = torch.zeros((shp_output[0], shp_output[1]), device=output.device)
            fp = torch.zeros((shp_output[0], shp_output[1]), device=output.device)
            fn = torch.zeros((shp_output[0], shp_output[1]), device=output.device)

            for b in range(shp_output[0]):
                for r in valid_regions[b]:
                    labels_in_region = MultiTalent_regions[r]
                    output_idx = MultiTalent_region_output_idx_mapping[r]

                    gt_b_c = target[b] == labels_in_region[0]
                    for l in labels_in_region[1:]:
                        gt_b_c = torch.bitwise_or(gt_b_c, (target[b] == l))
                    gt_b_c = gt_b_c.float()

                    tp[b, output_idx] += torch.sum(out_sigmoid[b, output_idx] * gt_b_c)
                    fp[b, output_idx] += torch.sum(
                        out_sigmoid[b, output_idx] * (1 - gt_b_c)
                    )
                    fn[b, output_idx] += torch.sum(
                        (1 - out_sigmoid[b, output_idx]) * gt_b_c
                    )

            tp = awesome_allgather_function.apply(tp)
            fp = awesome_allgather_function.apply(fp)
            fn = awesome_allgather_function.apply(fn)

            tp_hard = tp.detach().cpu().numpy()
            fp_hard = fp.detach().cpu().numpy()
            fn_hard = fn.detach().cpu().numpy()

            self.online_eval_foreground_dc.append(
                list((2 * tp_hard) / (2 * tp_hard + fp_hard + fn_hard + 1e-8))
            )
            self.online_eval_tp.append(list(tp_hard.sum(0)))
            self.online_eval_fp.append(list(fp_hard.sum(0)))
            self.online_eval_fn.append(list(fn_hard.sum(0)))

    def finish_online_evaluation(self):
        self.online_eval_tp = np.sum(self.online_eval_tp, 0)
        self.online_eval_fp = np.sum(self.online_eval_fp, 0)
        self.online_eval_fn = np.sum(self.online_eval_fn, 0)

        global_dc_per_class = [
            l
            for l in [
                2 * i / (np.clip(2 * i + j + k, a_min=1e-8, a_max=None))
                for i, j, k in zip(
                    self.online_eval_tp, self.online_eval_fp, self.online_eval_fn
                )
            ]
            if not np.isnan(l).any()
        ]
        self.all_val_eval_metrics.append(np.mean(global_dc_per_class))

        self.print_to_log_file(
            "Average global foreground Dice:", str(global_dc_per_class)
        )
        self.print_to_log_file(
            "(interpret this as an estimate for the Dice of the different classes. This is not "
            "exact.)"
        )

        self.online_eval_foreground_dc = []
        self.online_eval_tp = []
        self.online_eval_fp = []
        self.online_eval_fn = []

    def do_split(self):
        keys = list(self.dataset.keys())
        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            tr_keys = val_keys = keys
        else:
            if not isfile(join(self.dataset_directory, "splits_custom.pkl")):

                # fold 0-4 are a 5-fold CV that re-uses the splits from the individual datasets (except Task017 and
                # Task046 because of stupid duplicates)
                fivefold = [{'train': [], 'val': []} for _ in range(5)]
                preprocessed_dir = preprocessing_output_dir
                for task_id in np.unique([int(i.split("_")[0]) for i in keys]):
                    if task_id != 46:
                        task_name = convert_id_to_task_name(task_id)
                        expected_splits_file = join(preprocessed_dir, task_name, 'splits_final.pkl')
                        splits_t = load_pickle(expected_splits_file)
                        for f in range(5):
                            fivefold[f]['train'] += ["%03.0d_" % task_id + i for i in splits_t[f]['train']]
                            fivefold[f]['val'] += ["%03.0d_" % task_id + i for i in splits_t[f]['val']]
                    else:
                        # for task 46 we ignore the predefined split. That is because we have images from Task17 train and
                        # test set in there. So what we need to do is
                        # 1) distribute the images from task46 that are also in task 17 according to the task17 split.
                        # 2) remove all the task17 test set images
                        # 3) distribute the remaining images randomly to the 5 folds (seeded, of course)
                        remaining_task_46_ids = [i for i in keys if
                                                 i.startswith("046_PAN")]  # this is just the new task46 cases
                        rs = np.random.RandomState(1234)
                        rs.shuffle(remaining_task_46_ids)

                        task17_splits = load_pickle(join(preprocessed_dir, convert_id_to_task_name(17),
                                                         'splits_final.pkl'))
                        for f in range(5):
                            fivefold[f]['train'] += ["%03.0d_" % 46 + i for i in task17_splits[f]['train']]
                            fivefold[f]['val'] += ["%03.0d_" % 46 + i for i in task17_splits[f]['val']]
                            selected_val = remaining_task_46_ids[f::5]
                            selected_train = [i for i in remaining_task_46_ids if i not in selected_val]
                            fivefold[f]['train'] += selected_train
                            fivefold[f]['val'] += selected_val

                custom_splits = []

                # there are no validation sets here anymore!!!! We essentially train on 'all' for the following

                # fold 5 is all cases except those from Task003
                all_but_3 = [i for i in keys if not i.startswith("003_")]
                all_3 = [i for i in keys if i.startswith("003_")]
                custom_splits.append({'train': all_but_3, 'val': all_but_3})

                # fold 6 is all cases except those from Task017. We also need to exclude cases from Task046 that originate
                # from Task017 (Task046 even contains test set images of Task017. Ouch.
                all_but_17 = [i for i in keys if not i.startswith("017_") and not i.startswith("046_img")]
                all_17 = [i for i in keys if i.startswith("017_") or i.startswith("046_img")]
                custom_splits.append({'train': all_but_17, 'val': all_but_17})

                # fold 7 is all cases except those from Task064
                all_but_64 = [i for i in keys if not i.startswith("064_")]
                all_64 = [i for i in keys if i.startswith("064_")]
                custom_splits.append({'train': all_but_64, 'val': all_but_64})

                # fold 8 uses all cases except those from Task010
                all_but_10 = [i for i in keys if not i.startswith("010_")]
                all_10 = [i for i in keys if i.startswith("010_")]
                custom_splits.append({'train': all_but_10, 'val': all_but_10})

                # fold 9 uses all cases except those from Task007
                all_but_07 = [i for i in keys if not i.startswith("007_")]
                all_07 = [i for i in keys if i.startswith("007_")]
                custom_splits.append({'train': all_but_07, 'val': all_but_07})

                # fold 10 uses all cases except those from Task055
                all_but_55 = [i for i in keys if not i.startswith("055_")]
                all_55 = [i for i in keys if i.startswith("055_")]
                custom_splits.append({'train': all_but_55, 'val': all_but_55})

                # fold 11 uses all cases except those from Task008
                all_but_08 = [i for i in keys if not i.startswith("008_")]
                all_08 = [i for i in keys if i.startswith("008_")]
                custom_splits.append({'train': all_but_08, 'val': all_but_08})

                splits = fivefold + custom_splits

                # save it for later use
                if self.local_rank == 0:
                    save_pickle(splits, join(self.dataset_directory, "splits_custom.pkl"))

            while not isfile(join(self.dataset_directory, "splits_custom.pkl")):
                sleep(0.01)

            # now select the desired split
            splits = load_pickle(join(self.dataset_directory, "splits_custom.pkl"))
            tr_keys = splits[self.fold]['train']
            val_keys = splits[self.fold]['val']

        tr_keys.sort()
        val_keys.sort()
        self.dataset_tr = OrderedDict()
        for i in tr_keys:
            if i in self.dataset.keys():
                self.dataset_tr[i] = self.dataset[i]
            else:
                self.print_to_log_file('Warning %s is not in preprocessed folder (might be intentional)' % i)

        self.dataset_val = OrderedDict()
        for i in val_keys:
            if i in self.dataset.keys():
                self.dataset_val[i] = self.dataset[i]
            else:
                self.print_to_log_file('Warning %s is not in preprocessed folder (might be intentional)' % i)

    def compute_loss(self, output, target, valid_regions):
        """
        instead of averaging the loss we sum it over all classes. This way, classes always contribute the same gradient
        irrespective of how many other classes are present in the batch (not image, we are doing batch dice!). This
        should improve the results for left and right adrenal gland which are difficult to detect and only appear in
        conjunction with 12 other labels

        Note that this will give very ugly looking loss curves!

        :param output:
        :param target:
        :param valid_regions:
        :return:
        """
        total_loss = None
        total_ce = None
        total_dc = None
        # bczxy
        for i in range(len(output)):
            net_output = output[i]
            net_target = target[i]

            shp_output = net_output.shape

            tp = torch.zeros((shp_output[0], shp_output[1]), device=net_output.device)
            fp = torch.zeros((shp_output[0], shp_output[1]), device=net_output.device)
            fn = torch.zeros((shp_output[0], shp_output[1]), device=net_output.device)

            ce_loss = None

            output_sigmoid = torch.sigmoid(net_output)

            for b in range(shp_output[0]):
                for r in valid_regions[b]:
                    labels_in_region = MultiTalent_regions[r]
                    output_idx = MultiTalent_region_output_idx_mapping[r]

                    with torch.no_grad():
                        gt_b_c = net_target[b] == labels_in_region[0]
                        for l in labels_in_region[1:]:
                            gt_b_c = torch.bitwise_or(gt_b_c, (net_target[b] == l))
                        gt_b_c = gt_b_c.float()

                    if ce_loss is None:
                        ce_loss = self.ce_loss(net_output[b, output_idx], gt_b_c[0])
                    else:
                        ce_loss += self.ce_loss(net_output[b, output_idx], gt_b_c[0])

                    tp[b, output_idx] += torch.sum(
                        output_sigmoid[b, output_idx] * gt_b_c
                    )
                    fp[b, output_idx] += torch.sum(
                        output_sigmoid[b, output_idx] * (1 - gt_b_c)
                    )
                    fn[b, output_idx] += torch.sum(
                        (1 - output_sigmoid[b, output_idx]) * gt_b_c
                    )

            if self.batch_dice:
                # for DDP we need to gather all nominator and denominator terms from all GPUS to do proper batch dice
                tp = awesome_allgather_function.apply(tp)
                fp = awesome_allgather_function.apply(fp)
                fn = awesome_allgather_function.apply(fn)

                tp = tp.sum(0, keepdim=True)
                fp = fp.sum(0, keepdim=True)
                fn = fn.sum(0, keepdim=True)
                dc = 2 * tp / (torch.clamp(2 * tp + fp + fn, min=1e-7))
                dc = dc.sum()
            else:
                dc = 2 * tp / (torch.clamp(2 * tp + fp + fn, min=1e-7))
                dc = dc.sum()

            if total_loss is None:
                total_loss = self.ds_loss_weights[i] * (ce_loss - dc)
            else:
                total_loss += self.ds_loss_weights[i] * (ce_loss - dc)
            if total_ce is None:
                total_ce = self.ds_loss_weights[i] * ce_loss
            else:
                total_ce += self.ds_loss_weights[i] * ce_loss
            if total_dc is None:
                total_dc = self.ds_loss_weights[i] * dc
            else:
                total_dc += self.ds_loss_weights[i] * dc
        return total_loss, total_ce, total_dc

    def get_basic_generators(self):
        self.load_dataset()
        self.do_split()

        keys = list(self.dataset_tr.keys())
        dataset_dentifiers = list(np.unique([i.split("_")[0] for i in keys]))
        num_cases_per_dataset = [
            len([i for i in keys if i.startswith(j + "_")]) for j in dataset_dentifiers
        ]
        probabilities = np.array(
            [
                1
                / (
                    num_cases_per_dataset[dataset_dentifiers.index(i.split("_")[0])]
                    ** 0.5
                )
                for i in keys
            ]
        )
        probabilities = probabilities / sum(probabilities)
        self.dataset_prob = {}
        self.print_to_log_file(
            "cases per datasset train:\n",
            list(zip(dataset_dentifiers, num_cases_per_dataset)),
        )
        self.print_to_log_file("probabilities per dataset:")
        for d in dataset_dentifiers:
            dataset_keys = [i for i in keys if i.startswith(d + "_")]
            p_per_case = probabilities[keys.index(dataset_keys[0])]
            p_per_dataset = p_per_case * len(dataset_keys)
            self.print_to_log_file(d, p_per_case, p_per_dataset)
            self.dataset_prob[d] = p_per_dataset

        keys_val = list(self.dataset_val.keys())
        dataset_dentifiers_val = list(np.unique([i.split("_")[0] for i in keys_val]))
        num_cases_per_dataset = [
            len([i for i in keys_val if i.startswith(j + "_")])
            for j in dataset_dentifiers_val
        ]
        probabilities_val = np.array(
            [
                1
                / (
                    num_cases_per_dataset[dataset_dentifiers_val.index(i.split("_")[0])]
                    ** 0.5
                )
                for i in keys_val
            ]
        )
        probabilities_val = probabilities_val / sum(probabilities_val)
        if self.threeD:
            dl_tr = DataLoader3D(
                self.dataset_tr,
                self.basic_generator_patch_size,
                self.patch_size,
                self.batch_size,
                False,
                oversample_foreground_percent=self.oversample_foreground_percent,
                pad_mode="constant",
                pad_sides=self.pad_all_sides,
                memmap_mode="r",
                sampling_probabilities=probabilities,
            )
            dl_val = DataLoader3D(
                self.dataset_val,
                self.patch_size,
                self.patch_size,
                self.batch_size,
                False,
                oversample_foreground_percent=self.oversample_foreground_percent,
                pad_mode="constant",
                pad_sides=self.pad_all_sides,
                memmap_mode="r",
                sampling_probabilities=probabilities_val,
            )
        else:
            raise NotImplementedError
        return dl_tr, dl_val

    def run_training(self):
        """
        if we run with -c then we need to set the correct lr for the first epoch, otherwise it will run the first
        continued epoch with self.initial_lr

        we also need to make sure deep supervision in the network is enabled for training, thus the wrapper
        :return:
        """
        if self.local_rank == 0:
            self.save_debug_information()

        if not torch.cuda.is_available():
            self.print_to_log_file(
                "WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY slow!"
            )

        self.maybe_update_lr(
            self.epoch
        )  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
        # want at the start of the training
        if isinstance(self.network, DDP):
            net = self.network.module
        else:
            net = self.network
        net.do_ds = net.decoder.deep_supervision  # I am stupid and I know it - lol
        ds = net.decoder.deep_supervision
        net.decoder.deep_supervision = True

        _ = self.tr_gen.next()
        _ = self.val_gen.next()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._maybe_init_amp()

        maybe_mkdir_p(self.output_folder)
        self.plot_network_architecture()

        if cudnn.benchmark and cudnn.deterministic:
            warn(
                "torch.backends.cudnn.deterministic is True indicating a deterministic training is desired. "
                "But torch.backends.cudnn.benchmark is True as well and this will prevent deterministic training! "
                "If you want deterministic then set benchmark=False"
            )

        if not self.was_initialized:
            self.initialize(True)

        while self.epoch < self.max_num_epochs:
            self.print_to_log_file("\nepoch: ", self.epoch)

            epoch_start_time = time()
            train_losses_epoch = []
            ce_losses = []
            dc_losses = []

            # train one epoch
            self.network.train()

            if self.use_progress_bar:
                with trange(self.num_batches_per_epoch) as tbar:
                    for b in tbar:
                        tbar.set_description(
                            "Epoch {}/{}".format(self.epoch + 1, self.max_num_epochs)
                        )

                        l, ce, dc = self.run_iteration(self.tr_gen, True)

                        tbar.set_postfix(loss=l)
                        train_losses_epoch.append(l)
                        ce_losses.append(ce)
                        dc_losses.append(dc)

            else:
                for _ in range(self.num_batches_per_epoch):
                    l, ce, dc = self.run_iteration(self.tr_gen, True)
                    train_losses_epoch.append(l)
                    ce_losses.append(ce)
                    dc_losses.append(dc)

            self.all_tr_losses.append(np.mean(train_losses_epoch))
            self.print_to_log_file("train loss : %.4f" % self.all_tr_losses[-1])
            self.print_to_log_file("train ce loss : %.4f" % np.mean(ce_losses))
            self.print_to_log_file("train dc loss : %.4f" % np.mean(dc_losses))

            with torch.no_grad():
                # validation with train=False
                self.network.eval()
                val_losses = []
                for b in range(self.num_val_batches_per_epoch):
                    l, ce, dc = self.run_iteration(self.val_gen, False, True)
                    val_losses.append(l)
                self.all_val_losses.append(np.mean(val_losses))
                self.print_to_log_file(
                    "validation loss: %.4f" % self.all_val_losses[-1]
                )

                if self.also_val_in_tr_mode:
                    self.network.train()
                    # validation with train=True
                    val_losses = []
                    for b in range(self.num_val_batches_per_epoch):
                        l, ce, dc = self.run_iteration(self.val_gen, False)
                        val_losses.append(l)
                    self.all_val_losses_tr_mode.append(np.mean(val_losses))
                    self.print_to_log_file(
                        "validation loss (train=True): %.4f"
                        % self.all_val_losses_tr_mode[-1]
                    )

            self.update_train_loss_MA()  # needed for lr scheduler and stopping of training

            continue_training = self.on_epoch_end()

            epoch_end_time = time()

            if not continue_training:
                # allows for early stopping
                break

            self.epoch += 1
            self.print_to_log_file(
                "This epoch took %f s\n" % (epoch_end_time - epoch_start_time)
            )

        self.epoch -= 1  # if we don't do this we can get a problem with loading model_final_checkpoint.

        if self.save_final_checkpoint:
            self.save_checkpoint(
                join(self.output_folder, "model_final_checkpoint.model")
            )

        if self.local_rank == 0:
            # now we can delete latest as it will be identical with final
            if isfile(join(self.output_folder, "model_latest.model")):
                os.remove(join(self.output_folder, "model_latest.model"))
            if isfile(join(self.output_folder, "model_latest.model.pkl")):
                os.remove(join(self.output_folder, "model_latest.model.pkl"))
        net.decoder.deep_supervision = ds


class MultiTalent_trainer_resenc_ddp_2000ep(
    MultiTalent_trainer_resenc_ddp
):
    def __init__(
        self,
        plans_file,
        fold,
        local_rank,
        output_folder=None,
        dataset_directory=None,
        batch_dice=True,
        stage=None,
        unpack_data=True,
        deterministic=True,
        distribute_batch_size=False,
        fp16=False,
    ):
        batch_dice = True
        super().__init__(
            plans_file=plans_file,
            fold=fold,
            local_rank=local_rank,
            output_folder=output_folder,
            dataset_directory=dataset_directory,
            batch_dice=batch_dice,
            stage=stage,
            unpack_data=unpack_data,
            deterministic=deterministic,
            fp16=fp16,
            distribute_batch_size=distribute_batch_size,
        )
        self.max_num_epochs = 2000
