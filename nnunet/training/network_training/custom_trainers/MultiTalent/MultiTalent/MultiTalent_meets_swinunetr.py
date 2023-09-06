import torch
import torch.nn as nn
from monai.networks.nets.swin_unetr import SwinUNETR as SwinUNETR_Orig
from nnunet.training.network_training.custom_trainers.MultiTalent.MultiTalent.MultiTalent_Trainer_DDP \
    import MultiTalent_trainer_ddp
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.dataset_conversion.Task100_MultiTalent import MultiTalent_regions, MultiTalent_region_output_idx_mapping
from nnunet.utilities.distributed import awesome_allgather_function
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import distributed
from nnunet.training.dataloading.dataset_loading import unpack_dataset
from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from nnunet.paths import preprocessing_output_dir
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
from collections import OrderedDict

# Wrapper using SegmentationNetwork object. To pass nnUNet internal check
class SwinUNETR(SwinUNETR_Orig, SegmentationNetwork):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Segmentation Network Params. Needed for the nnUNet evaluation pipeline
        self.conv_op = nn.Conv3d
        self.inference_apply_nonlin = softmax_helper
        self.input_shape_must_be_divisible_by = 16  # just some random val 2**5
        self.num_classes = kwargs['out_channels']
        self.do_ds = False


class MultiTalent_tainer_SwinUNETR_ddp_adam(MultiTalent_trainer_ddp):
    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs)
        self.initial_lr = 5e-4

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.AdamW(self.network.parameters(),
                                            self.initial_lr,
                                           weight_decay=self.weight_decay,
                                           eps=1e-4  # 1e-8 might cause nans in fp16
                                           )
        self.lr_scheduler = None


    def initialize_network(self):
        """
        changed deep supervision to False
        :return:
        """
        self.network = SwinUNETR(img_size=(96, 192, 192),
                                 in_channels=self.num_input_channels,
                                 out_channels=self.num_classes,
                                 feature_size=48,
                                 use_checkpoint=False,
                                 )
        # self.network = Generic_UNet(self.num_input_channels, self.base_num_features, self.num_classes,
        #                             len(self.net_num_pool_op_kernel_sizes),
        #                             self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
        #                             net_nonlin, net_nonlin_kwargs, False, False, lambda x: x, InitWeights_He(1e-2),
        #                             self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = nn.Sigmoid()

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
        net_output = output
        net_target = target

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
                    gt_b_c = (net_target[b] == labels_in_region[0])
                    for l in labels_in_region[1:]:
                        gt_b_c = torch.bitwise_or(gt_b_c, (net_target[b] == l))
                    gt_b_c = gt_b_c.float()

                if ce_loss is None:
                    ce_loss = self.ce_loss(net_output[b, output_idx], gt_b_c[0])
                else:
                    ce_loss += self.ce_loss(net_output[b, output_idx], gt_b_c[0])

                tp[b, output_idx] += torch.sum(output_sigmoid[b, output_idx] * gt_b_c)
                fp[b, output_idx] += torch.sum(output_sigmoid[b, output_idx] * (1 - gt_b_c))
                fn[b, output_idx] += torch.sum((1 - output_sigmoid[b, output_idx]) * gt_b_c)

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

        total_loss = ce_loss - dc

        return total_loss, ce_loss, dc

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
                self.tr_gen, self.val_gen = get_moreDA_augmentation(self.dl_tr, self.dl_val,
                                                                    self.data_aug_params[
                                                                        'patch_size_for_spatialtransform'],
                                                                    self.data_aug_params,
                                                                    seeds_train=seeds_train,
                                                                    seeds_val=seeds_val,
                                                                    pin_memory=self.pin_memory,
                                                                    order_seg=0)
                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()
            self.network = DDP(self.network, device_ids=[self.local_rank])

        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

        self.regions_class_order = list(range(self.num_classes))

    def run_online_evaluation(self, output, target, valid_regions):
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

                    gt_b_c = (target[b] == labels_in_region[0])
                    for l in labels_in_region[1:]:
                        gt_b_c = torch.bitwise_or(gt_b_c, (target[b] == l))
                    gt_b_c = gt_b_c.float()

                    tp[b, output_idx] += torch.sum(out_sigmoid[b, output_idx] * gt_b_c)
                    fp[b, output_idx] += torch.sum(out_sigmoid[b, output_idx] * (1 - gt_b_c))
                    fn[b, output_idx] += torch.sum((1 - out_sigmoid[b, output_idx]) * gt_b_c)

            tp = awesome_allgather_function.apply(tp)
            fp = awesome_allgather_function.apply(fp)
            fn = awesome_allgather_function.apply(fn)

            tp_hard = tp.detach().cpu().numpy()
            fp_hard = fp.detach().cpu().numpy()
            fn_hard = fn.detach().cpu().numpy()

            self.online_eval_foreground_dc.append(list((2 * tp_hard) / (2 * tp_hard + fp_hard + fn_hard + 1e-8)))
            self.online_eval_tp.append(list(tp_hard.sum(0)))
            self.online_eval_fp.append(list(fp_hard.sum(0)))
            self.online_eval_fn.append(list(fn_hard.sum(0)))





class MultiTalent_tainer_SwinUNETR_ddp_adam_2000ep(MultiTalent_tainer_SwinUNETR_ddp_adam):
    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs)
        self.max_num_epochs = 2000

