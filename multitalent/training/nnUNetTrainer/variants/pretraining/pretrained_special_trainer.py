
from multitalent.training.loss.focal_loss import FocalLoss_Ori, Focal_and_DC_Loss
from multitalent.training.loss.recall_loss import Recall_and_DC_Loss
from multitalent.training.nnUNetTrainer.variants.pretraining.multiple_inputs_warmup import nnUNetTrainer_warmupnet, \
    nnUNetTrainer_warmupnet_nosmooth
from multitalent.configuration import ANISO_THRESHOLD
from typing import List, Union, Tuple
from multitalent.training.data_augmentation.compute_initial_patch_size import get_patch_size
import numpy as np

from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.color_transforms import BrightnessTransform, ContrastAugmentationTransform, \
    GammaTransform
from batchgenerators.transforms.local_transforms import BrightnessGradientAdditiveTransform, LocalGammaTransform
from batchgenerators.transforms.noise_transforms import MedianFilterTransform, GaussianBlurTransform, \
    GaussianNoiseTransform, BlankRectangleTransform, SharpeningTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, Rot90Transform, TransposeAxesTransform, \
    MirrorTransform
from batchgenerators.transforms.utility_transforms import OneOfTransform, RemoveLabelTransform, RenameTransform, \
    NumpyToTensor
from batchgeneratorsv2.helpers.scalar_type import RandomScalar

from multitalent.training.data_augmentation.custom_transforms.cascade_transforms import MoveSegAsOneHotToData, \
    ApplyRandomBinaryOperatorTransform, RemoveRandomConnectedComponentFromOneHotEncodingTransform
from multitalent.training.data_augmentation.custom_transforms.deep_supervision_donwsampling import \
    DownsampleSegForDSTransform2
from multitalent.training.data_augmentation.custom_transforms.masking import MaskTransform
from multitalent.training.data_augmentation.custom_transforms.region_based_training import \
    ConvertSegmentationToRegionsTransform
from multitalent.training.data_augmentation.custom_transforms.transforms_for_dummy_2d import Convert3DTo2DTransform, \
    Convert2DTo3DTransform
from multitalent.training.dataloading.data_loader_2d import nnUNetDataLoader2D
from multitalent.training.dataloading.data_loader_3d import nnUNetDataLoader3D

from multitalent.training.loss.deep_supervision import DeepSupervisionWrapper
import torch
from multitalent.utilities.default_n_proc_DA import get_allowed_n_proc_DA


def _brightnessadditive_localgamma_transform_scale(x, y):
    return np.exp(np.random.uniform(np.log(x[y] // 6), np.log(x[y])))

def _brightness_gradient_additive_max_strength(_x, _y):
    return np.random.uniform(-5, -1) if np.random.uniform() < 0.5 else np.random.uniform(1, 5)


def _local_gamma_gamma():
    return np.random.uniform(0.01, 0.8) if np.random.uniform() < 0.5 else np.random.uniform(1.5, 4)


class nnUNetTrainerFocalLoss_warmupnet(nnUNetTrainer_warmupnet):
    def _build_loss(self):
        assert not self.label_manager.has_regions
        loss = FocalLoss_Ori(self.label_manager.num_segmentation_heads, ignore_index=self.label_manager.ignore_label)

        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)
        return loss

class nnUNetTrainerRecall_warmupnet(nnUNetTrainer_warmupnet):
    def _build_loss(self):
        assert not self.label_manager.has_regions
        loss = Recall_and_DC_Loss({'batch_dice': self.configuration_manager.batch_dice,
                                  'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, ignore_label=self.label_manager.ignore_label)

        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)
        return loss

class nnUNetTrainerRecall_warmupnet_nosmooth(nnUNetTrainer_warmupnet):
    def _build_loss(self):
        assert not self.label_manager.has_regions
        loss = Recall_and_DC_Loss({'batch_dice': self.configuration_manager.batch_dice,
                                  'smooth': 0, 'do_bg': False, 'ddp': self.is_ddp}, ignore_label=self.label_manager.ignore_label)

        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)
        return loss
class nnUNetTrainerFocalandDiceLoss_warmupnet(nnUNetTrainer_warmupnet):
    def _build_loss(self):
        assert not self.label_manager.has_regions, "region-based training not supported here"
        # loss = FocalLoss_Ori(num_class=self.label_manager.num_segmentation_heads, ignore_index=self.label_manager.ignore_label)
        loss = Focal_and_DC_Loss({'batch_dice': self.configuration_manager.batch_dice,
                                  'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
                                 {'num_class': self.label_manager.num_segmentation_heads,
                                  'ignore_index': self.label_manager.ignore_label},
                                 ignore_label=self.label_manager.ignore_label)

        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)
        return loss

class nnUNetTrainerDA5_warmupnet(nnUNetTrainer_warmupnet):
    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)
        # todo rotation should be defined dynamically based on patch size (more isotropic patch sizes = more rotation)
        if dim == 2:
            do_dummy_2d_data_aug = False
            # todo revisit this parametrization
            if max(patch_size) / min(patch_size) > 1.5:
                rotation_for_DA = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            else:
                rotation_for_DA = (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi)
            mirror_axes = (0, 1)
        elif dim == 3:
            # todo this is not ideal. We could also have patch_size (64, 16, 128) in which case a full 180deg 2d rot would be bad
            # order of the axes is determined by spacing, not image size
            do_dummy_2d_data_aug = (max(patch_size) / patch_size[0]) > ANISO_THRESHOLD
            if do_dummy_2d_data_aug:
                # why do we rotate 180 deg here all the time? We should also restrict it
                rotation_for_DA = (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi)
            else:
                rotation_for_DA = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            mirror_axes = (0, 1, 2)
        else:
            raise RuntimeError()

        # todo this function is stupid. It doesn't even use the correct scale range (we keep things as they were in the
        #  old nnunet for now)
        initial_patch_size = get_patch_size(patch_size[-dim:],
                                            rotation_for_DA,
                                            rotation_for_DA,
                                            rotation_for_DA,
                                            (0.7, 1.43))
        if do_dummy_2d_data_aug:
            initial_patch_size[0] = patch_size[0]

        self.print_to_log_file(f'do_dummy_2d_data_aug: {do_dummy_2d_data_aug}')
        self.inference_allowed_mirroring_axes = mirror_axes

        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

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
    ) -> AbstractTransform:
        matching_axes = np.array([sum([i == j for j in patch_size]) for i in patch_size])
        valid_axes = list(np.where(matching_axes == np.max(matching_axes))[0])

        tr_transforms = []
        tr_transforms.append(RenameTransform('target', 'seg', True))

        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            tr_transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None

        tr_transforms.append(
            SpatialTransform(
                patch_size_spatial,
                patch_center_dist_from_border=None,
                do_elastic_deform=False,
                do_rotation=True,
                angle_x=rotation_for_DA,
                angle_y=rotation_for_DA,
                angle_z=rotation_for_DA,
                p_rot_per_axis=0.5,
                do_scale=True,
                scale=(0.7, 1.43),
                border_mode_data="constant",
                border_cval_data=0,
                order_data=3,
                border_mode_seg="constant",
                border_cval_seg=-1,
                order_seg=1,
                random_crop=False,
                p_el_per_sample=0.2,
                p_scale_per_sample=0.2,
                p_rot_per_sample=0.4,
                independent_scale_for_each_axis=True,
            )
        )

        if do_dummy_2d_data_aug:
            tr_transforms.append(Convert2DTo3DTransform())

        if np.any(matching_axes > 1):
            tr_transforms.append(
                Rot90Transform(
                    (0, 1, 2, 3), axes=valid_axes, data_key='data', label_key='seg', p_per_sample=0.5
                ),
            )

        if np.any(matching_axes > 1):
            tr_transforms.append(
                TransposeAxesTransform(valid_axes, data_key='data', label_key='seg', p_per_sample=0.5)
            )

        tr_transforms.append(OneOfTransform([
            MedianFilterTransform(
                (2, 8),
                same_for_each_channel=False,
                p_per_sample=0.2,
                p_per_channel=0.5
            ),
            GaussianBlurTransform((0.3, 1.5),
                                  different_sigma_per_channel=True,
                                  p_per_sample=0.2,
                                  p_per_channel=0.5)
        ]))

        tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))

        tr_transforms.append(BrightnessTransform(0,
                                                 0.5,
                                                 per_channel=True,
                                                 p_per_sample=0.1,
                                                 p_per_channel=0.5
                                                 )
                             )

        tr_transforms.append(OneOfTransform(
            [
                ContrastAugmentationTransform(
                    contrast_range=(0.5, 2),
                    preserve_range=True,
                    per_channel=True,
                    data_key='data',
                    p_per_sample=0.2,
                    p_per_channel=0.5
                ),
                ContrastAugmentationTransform(
                    contrast_range=(0.5, 2),
                    preserve_range=False,
                    per_channel=True,
                    data_key='data',
                    p_per_sample=0.2,
                    p_per_channel=0.5
                ),
            ]
        ))

        tr_transforms.append(
            SimulateLowResolutionTransform(zoom_range=(0.25, 1),
                                           per_channel=True,
                                           p_per_channel=0.5,
                                           order_downsample=0,
                                           order_upsample=3,
                                           p_per_sample=0.15,
                                           ignore_axes=ignore_axes
                                           )
        )

        tr_transforms.append(
            GammaTransform((0.7, 1.5), invert_image=True, per_channel=True, retain_stats=True, p_per_sample=0.1))
        tr_transforms.append(
            GammaTransform((0.7, 1.5), invert_image=True, per_channel=True, retain_stats=True, p_per_sample=0.1))

        if mirror_axes is not None and len(mirror_axes) > 0:
            tr_transforms.append(MirrorTransform(mirror_axes))

        tr_transforms.append(
            BlankRectangleTransform([[max(1, p // 10), p // 3] for p in patch_size],
                                    rectangle_value=np.mean,
                                    num_rectangles=(1, 5),
                                    force_square=False,
                                    p_per_sample=0.4,
                                    p_per_channel=0.5
                                    )
        )

        tr_transforms.append(
            BrightnessGradientAdditiveTransform(
                _brightnessadditive_localgamma_transform_scale,
                (-0.5, 1.5),
                max_strength=_brightness_gradient_additive_max_strength,
                mean_centered=False,
                same_for_all_channels=False,
                p_per_sample=0.3,
                p_per_channel=0.5
            )
        )

        tr_transforms.append(
            LocalGammaTransform(
                _brightnessadditive_localgamma_transform_scale,
                (-0.5, 1.5),
                _local_gamma_gamma,
                same_for_all_channels=False,
                p_per_sample=0.3,
                p_per_channel=0.5
            )
        )

        tr_transforms.append(
            SharpeningTransform(
                strength=(0.1, 1),
                same_for_each_channel=False,
                p_per_sample=0.2,
                p_per_channel=0.5
            )
        )

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            tr_transforms.append(MaskTransform([i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                                               mask_idx_in_seg=0, set_outside_to=0))

        tr_transforms.append(RemoveLabelTransform(-1, 0))

        if is_cascaded:
            if ignore_label is not None:
                raise NotImplementedError('ignore label not yet supported in cascade')
            assert foreground_labels is not None, 'We need all_labels for cascade augmentations'
            use_labels = [i for i in foreground_labels if i != 0]
            tr_transforms.append(MoveSegAsOneHotToData(1, use_labels, 'seg', 'data'))
            tr_transforms.append(ApplyRandomBinaryOperatorTransform(
                channel_idx=list(range(-len(use_labels), 0)),
                p_per_sample=0.4,
                key="data",
                strel_size=(1, 8),
                p_per_label=1))
            tr_transforms.append(
                RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                    channel_idx=list(range(-len(use_labels), 0)),
                    key="data",
                    p_per_sample=0.2,
                    fill_with_other_class_p=0,
                    dont_do_if_covers_more_than_x_percent=0.15))

        tr_transforms.append(RenameTransform('seg', 'target', True))

        if regions is not None:
            # the ignore label must also be converted
            tr_transforms.append(ConvertSegmentationToRegionsTransform(list(regions) + [ignore_label]
                                                                       if ignore_label is not None else regions,
                                                                       'target', 'target'))

        if deep_supervision_scales is not None:
            tr_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key='target',
                                                              output_key='target'))
        tr_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
        tr_transforms = Compose(tr_transforms)
        return tr_transforms

    @staticmethod
    def get_validation_transforms(
            deep_supervision_scales: Union[List, Tuple, None],
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> AbstractTransform:
        val_transforms = []
        val_transforms.append(RenameTransform('target', 'seg', True))
        val_transforms.append(RemoveLabelTransform(-1, 0))

        if is_cascaded:
            val_transforms.append(MoveSegAsOneHotToData(1, foreground_labels, 'seg', 'data'))

        val_transforms.append(RenameTransform('seg', 'target', True))

        if regions is not None:
            # the ignore label must also be converted
            val_transforms.append(ConvertSegmentationToRegionsTransform(list(regions) + [ignore_label]
                                                                        if ignore_label is not None else regions,
                                                                        'target', 'target'))

        if deep_supervision_scales is not None:
            val_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key='target',
                                                               output_key='target'))

        val_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
        val_transforms = Compose(val_transforms)
        return val_transforms

    def get_dataloaders(self):
        # we use the patch size to determine whether we need 2D or 3D dataloaders. We also use it to determine whether
        # we need to use dummy 2D augmentation (in case of 3D training) and what our initial patch size should be
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)

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
        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        # validation pipeline
        val_transforms = self.get_validation_transforms(deep_supervision_scales,
                                                        is_cascaded=self.is_cascaded,
                                                        foreground_labels=self.label_manager.foreground_labels,
                                                        regions=self.label_manager.foreground_regions if
                                                        self.label_manager.has_regions else None,
                                                        ignore_label=self.label_manager.ignore_label)

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        # we set transforms=None because this trainer still uses batchgenerators which expects transforms to be passed to
        if dim == 2:
            dl_tr = nnUNetDataLoader2D(dataset_tr, self.batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=None, pad_sides=None, transforms=None)
            dl_val = nnUNetDataLoader2D(dataset_val, self.batch_size,
                                        self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size,
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        sampling_probabilities=None, pad_sides=None, transforms=None)
        else:
            dl_tr = nnUNetDataLoader3D(dataset_tr, self.batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=None, pad_sides=None, transforms=None)
            dl_val = nnUNetDataLoader3D(dataset_val, self.batch_size,
                                        self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size,
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        sampling_probabilities=None, pad_sides=None, transforms=None)

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, tr_transforms)
            mt_gen_val = SingleThreadedAugmenter(dl_val, val_transforms)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(data_loader=dl_tr, transform=tr_transforms,
                                                        num_processes=allowed_num_processes, num_cached=6, seeds=None,
                                                        pin_memory=self.device.type == 'cuda', wait_time=0.02)
            mt_gen_val = NonDetMultiThreadedAugmenter(data_loader=dl_val,
                                                      transform=val_transforms,
                                                      num_processes=max(1, allowed_num_processes // 2),
                                                      num_cached=3, seeds=None, pin_memory=self.device.type == 'cuda',
                                                      wait_time=0.02)
        # # let's get this party started
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val

class nnUNetTrainerRecallDA5_warmupnet(nnUNetTrainerDA5_warmupnet):
    def _build_loss(self):
        assert not self.label_manager.has_regions
        loss = Recall_and_DC_Loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
                                  ignore_label=self.label_manager.ignore_label)

        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)
        return loss

class nnUNetTrainer_warmupnet_100ep_lr1e3(nnUNetTrainer_warmupnet):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset =True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 100
        self.initial_lr = 1e-3

class nnUNetTrainer_warmupnet_100ep_lr1e4(nnUNetTrainer_warmupnet):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset =True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 100
        self.initial_lr = 1e-4
class nnUNetTrainer_warmupnet_100ep_lr1e4_nosmooth(nnUNetTrainer_warmupnet_nosmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset =True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 100
        self.initial_lr = 1e-4
class nnUNetTrainer_warmupnet_100ep_lr1e5(nnUNetTrainer_warmupnet):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset =True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 100
        self.initial_lr = 1e-5

class nnUNetTrainer_warmupnet_100ep_lr1e3_withhead(nnUNetTrainer_warmupnet):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset =True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 100 + self.warmup_duration_whole_net
        self.initial_lr = 1e-3

class nnUNetTrainer_warmupnet_200ep_lr1e3_withhead(nnUNetTrainer_warmupnet):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset =True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 200 + self.warmup_duration_whole_net
        self.initial_lr = 1e-3

class nnUNetTrainer_warmupnet_100ep_lr1e4_withhead(nnUNetTrainer_warmupnet):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset =True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 1000 + self.warmup_duration_whole_net
        self.initial_lr = 1e-4

class nnUNetTrainer_warmupnet_100ep_lr1e3_withhead_nosmooth(nnUNetTrainer_warmupnet_nosmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset =True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 100 + self.warmup_duration_whole_net
        self.initial_lr = 1e-3

class nnUNetTrainer_warmupnet_100ep_lr1e4_withhead_nosmooth(nnUNetTrainer_warmupnet_nosmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset =True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 100 + self.warmup_duration_whole_net
        self.initial_lr = 1e-4
class nnUNetTrainer_warmupnet_100ep_lr1e5_withhead(nnUNetTrainer_warmupnet):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset =True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 1000 + self.warmup_duration_whole_net
        self.initial_lr = 1e-5

class nnUNetTrainer_warmupnet_100ep_recall(nnUNetTrainerRecall_warmupnet):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset =True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 100 + self.warmup_duration_whole_net


class nnUNetTrainer_warmupnet_100ep_lr1e3_recall(nnUNetTrainerRecall_warmupnet):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset =True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 100 + self.warmup_duration_whole_net
        self.initial_lr = 1e-3
class nnUNetTrainer_warmupnet_100ep_lr1e4_withhead_recall(nnUNetTrainerRecall_warmupnet):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset =True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 100 + self.warmup_duration_whole_net
        self.initial_lr = 1e-4
class nnUNetTrainer_warmupnet_100ep_lr1e3_withhead_recall(nnUNetTrainerRecall_warmupnet):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset =True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 100 + self.warmup_duration_whole_net
        self.initial_lr = 1e-3
class nnUNetTrainer_warmupnet_100ep_lr1e5_withhead_recall(nnUNetTrainerRecall_warmupnet):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset =True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 1000 + self.warmup_duration_whole_net
        self.initial_lr = 1e-5

'''
bsub -gpu num=1:j_exclusive=yes:gmem=20G -q gpu -R tensorcore -o /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs/%J.log -e /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs//%J.err -L /bin/bash "source ~/.bashrc_v2challenges   &&  multitalent_train 402 3d_fullres 0 -tr nnUNetTrainer_warmupnet_100ep_lr1e3_recall -p nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/multitalent/nnUNet_trained_models/Dataset626_mbhmultitalentnopseudo/MultiTalent_trainer__nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT__3d_fullres/fold_0/checkpoint_final.pth -stemid 422"
bsub -gpu num=1:j_exclusive=yes:gmem=20G -q gpu -R tensorcore -o /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs/%J.log -e /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs//%J.err -L /bin/bash "source ~/.bashrc_v2challenges   &&  multitalent_train 402 3d_fullres 1 -tr nnUNetTrainer_warmupnet_100ep_lr1e3_recall -p nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/multitalent/nnUNet_trained_models/Dataset626_mbhmultitalentnopseudo/MultiTalent_trainer__nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT__3d_fullres/fold_1/checkpoint_final.pth -stemid 422"
bsub -gpu num=1:j_exclusive=yes:gmem=20G -q gpu -R tensorcore -o /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs/%J.log -e /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs//%J.err -L /bin/bash "source ~/.bashrc_v2challenges   &&  multitalent_train 402 3d_fullres 2 -tr nnUNetTrainer_warmupnet_100ep_lr1e3_recall -p nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/multitalent/nnUNet_trained_models/Dataset626_mbhmultitalentnopseudo/MultiTalent_trainer__nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT__3d_fullres/fold_2/checkpoint_final.pth -stemid 422"
bsub -gpu num=1:j_exclusive=yes:gmem=20G -q gpu -R tensorcore -o /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs/%J.log -e /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs//%J.err -L /bin/bash "source ~/.bashrc_v2challenges   &&  multitalent_train 402 3d_fullres 3 -tr nnUNetTrainer_warmupnet_100ep_lr1e3_recall -p nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/multitalent/nnUNet_trained_models/Dataset626_mbhmultitalentnopseudo/MultiTalent_trainer__nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT__3d_fullres/fold_3/checkpoint_final.pth -stemid 422"
bsub -gpu num=1:j_exclusive=yes:gmem=20G -q gpu -R tensorcore -o /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs/%J.log -e /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs//%J.err -L /bin/bash "source ~/.bashrc_v2challenges   &&  multitalent_train 402 3d_fullres 4 -tr nnUNetTrainer_warmupnet_100ep_lr1e3_recall -p nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/multitalent/nnUNet_trained_models/Dataset626_mbhmultitalentnopseudo/MultiTalent_trainer__nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT__3d_fullres/fold_4/checkpoint_final.pth -stemid 422"

bsub -gpu num=1:j_exclusive=yes:gmem=20G -q gpu -R tensorcore -o /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs/%J.log -e /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs//%J.err -L /bin/bash "source ~/.bashrc_v2challenges   &&  multitalent_train 402 3d_fullres 0 -tr nnUNetTrainer_warmupnet_100ep_recall -p nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/multitalent/nnUNet_trained_models/Dataset626_mbhmultitalentnopseudo/MultiTalent_trainer__nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT__3d_fullres/fold_0/checkpoint_final.pth -stemid 422"
bsub -gpu num=1:j_exclusive=yes:gmem=20G -q gpu -R tensorcore -o /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs/%J.log -e /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs//%J.err -L /bin/bash "source ~/.bashrc_v2challenges   &&  multitalent_train 402 3d_fullres 1 -tr nnUNetTrainer_warmupnet_100ep_recall -p nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/multitalent/nnUNet_trained_models/Dataset626_mbhmultitalentnopseudo/MultiTalent_trainer__nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT__3d_fullres/fold_1/checkpoint_final.pth -stemid 422"
bsub -gpu num=1:j_exclusive=yes:gmem=20G -q gpu -R tensorcore -o /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs/%J.log -e /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs//%J.err -L /bin/bash "source ~/.bashrc_v2challenges   &&  multitalent_train 402 3d_fullres 2 -tr nnUNetTrainer_warmupnet_100ep_recall -p nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/multitalent/nnUNet_trained_models/Dataset626_mbhmultitalentnopseudo/MultiTalent_trainer__nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT__3d_fullres/fold_2/checkpoint_final.pth -stemid 422"
bsub -gpu num=1:j_exclusive=yes:gmem=20G -q gpu -R tensorcore -o /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs/%J.log -e /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs//%J.err -L /bin/bash "source ~/.bashrc_v2challenges   &&  multitalent_train 402 3d_fullres 3 -tr nnUNetTrainer_warmupnet_100ep_recall -p nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/multitalent/nnUNet_trained_models/Dataset626_mbhmultitalentnopseudo/MultiTalent_trainer__nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT__3d_fullres/fold_3/checkpoint_final.pth -stemid 422"
bsub -gpu num=1:j_exclusive=yes:gmem=20G -q gpu -R tensorcore -o /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs/%J.log -e /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs//%J.err -L /bin/bash "source ~/.bashrc_v2challenges   &&  multitalent_train 402 3d_fullres 4 -tr nnUNetTrainer_warmupnet_100ep_recall -p nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/multitalent/nnUNet_trained_models/Dataset626_mbhmultitalentnopseudo/MultiTalent_trainer__nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT__3d_fullres/fold_4/checkpoint_final.pth -stemid 422"

bsub -gpu num=1:j_exclusive=yes:gmem=39G -q gpu-debian -R tensorcore -o /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs/%J.log -e /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs//%J.err ". ~/.bashrc_v2challenges_debian.sh  &&  multitalent_train 402 3d_fullres 0 -tr nnUNetTrainer_warmupnet_200ep_lr1e3_withhead -p nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT_big -pretrained_weights /omics/groups/OE0441/E132-Projekte/Projects/2024_Ulrich_collection/challenge_results/Dataset422_MBH_SEG_pseudo/pretrained/Dataset617_nativect__MultiTalent_trainer_4000ep__nnUNetResEncUNetL1x1x1_Plans_bs24__3d_fullres__fold_all/nnUNetTrainer_warmupnet_1e3_nosmooth__nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT__3d_fullres/fold_0/checkpoint_best.pth"
bsub -gpu num=1:j_exclusive=yes:gmem=39G -q gpu-debian -R tensorcore -o /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs/%J.log -e /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs//%J.err ". ~/.bashrc_v2challenges_debian.sh  &&  multitalent_train 402 3d_fullres 1 -tr nnUNetTrainer_warmupnet_200ep_lr1e3_withhead -p nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT_big -pretrained_weights /omics/groups/OE0441/E132-Projekte/Projects/2024_Ulrich_collection/challenge_results/Dataset422_MBH_SEG_pseudo/pretrained/Dataset617_nativect__MultiTalent_trainer_4000ep__nnUNetResEncUNetL1x1x1_Plans_bs24__3d_fullres__fold_all/nnUNetTrainer_warmupnet_1e3_nosmooth__nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT__3d_fullres/fold_1/checkpoint_best.pth"
bsub -gpu num=1:j_exclusive=yes:gmem=39G -q gpu-debian -R tensorcore -o /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs/%J.log -e /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs//%J.err ". ~/.bashrc_v2challenges_debian.sh  &&  multitalent_train 402 3d_fullres 2 -tr nnUNetTrainer_warmupnet_200ep_lr1e3_withhead -p nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT_big -pretrained_weights /omics/groups/OE0441/E132-Projekte/Projects/2024_Ulrich_collection/challenge_results/Dataset422_MBH_SEG_pseudo/pretrained/Dataset617_nativect__MultiTalent_trainer_4000ep__nnUNetResEncUNetL1x1x1_Plans_bs24__3d_fullres__fold_all/nnUNetTrainer_warmupnet_1e3_nosmooth__nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT__3d_fullres/fold_2/checkpoint_best.pth"
bsub -gpu num=1:j_exclusive=yes:gmem=39G -q gpu-debian -R tensorcore -o /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs/%J.log -e /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs//%J.err ". ~/.bashrc_v2challenges_debian.sh  &&  multitalent_train 402 3d_fullres 3 -tr nnUNetTrainer_warmupnet_200ep_lr1e3_withhead -p nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT_big -pretrained_weights /omics/groups/OE0441/E132-Projekte/Projects/2024_Ulrich_collection/challenge_results/Dataset422_MBH_SEG_pseudo/pretrained/Dataset617_nativect__MultiTalent_trainer_4000ep__nnUNetResEncUNetL1x1x1_Plans_bs24__3d_fullres__fold_all/nnUNetTrainer_warmupnet_1e3_nosmooth__nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT__3d_fullres/fold_3/checkpoint_best.pth"
bsub -gpu num=1:j_exclusive=yes:gmem=39G -q gpu-debian -R tensorcore -o /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs/%J.log -e /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs//%J.err ". ~/.bashrc_v2challenges_debian.sh  &&  multitalent_train 402 3d_fullres 4 -tr nnUNetTrainer_warmupnet_200ep_lr1e3_withhead -p nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT_big -pretrained_weights /omics/groups/OE0441/E132-Projekte/Projects/2024_Ulrich_collection/challenge_results/Dataset422_MBH_SEG_pseudo/pretrained/Dataset617_nativect__MultiTalent_trainer_4000ep__nnUNetResEncUNetL1x1x1_Plans_bs24__3d_fullres__fold_all/nnUNetTrainer_warmupnet_1e3_nosmooth__nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT__3d_fullres/fold_4/checkpoint_best.pth"

bsub -gpu num=1:j_exclusive=yes:gmem=20G -q gpu -R tensorcore -o /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs/%J.log -e /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs//%J.err -L /bin/bash "source ~/.bashrc_v2challenges   &&  multitalent_train 402 3d_fullres 0 -tr nnUNetTrainer_warmupnet_100ep_lr1e4_withhead_nosmooth -p nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/multitalent/nnUNet_trained_models/Dataset422_MBH_SEG_pseudo/pretrained/Dataset617_nativect__MultiTalent_trainer_4000ep__nnUNetResEncUNetL1x1x1_Plans_bs24__3d_fullres__fold_all/nnUNetTrainer_warmupnet_1e3_nosmooth__nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT__3d_fullres/fold_0/checkpoint_best.pth"
bsub -gpu num=1:j_exclusive=yes:gmem=20G -q gpu -R tensorcore -o /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs/%J.log -e /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs//%J.err -L /bin/bash "source ~/.bashrc_v2challenges   &&  multitalent_train 402 3d_fullres 1 -tr nnUNetTrainer_warmupnet_100ep_lr1e4_withhead_nosmooth -p nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/multitalent/nnUNet_trained_models/Dataset422_MBH_SEG_pseudo/pretrained/Dataset617_nativect__MultiTalent_trainer_4000ep__nnUNetResEncUNetL1x1x1_Plans_bs24__3d_fullres__fold_all/nnUNetTrainer_warmupnet_1e3_nosmooth__nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT__3d_fullres/fold_1/checkpoint_best.pth"
bsub -gpu num=1:j_exclusive=yes:gmem=20G -q gpu -R tensorcore -o /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs/%J.log -e /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs//%J.err -L /bin/bash "source ~/.bashrc_v2challenges   &&  multitalent_train 402 3d_fullres 2 -tr nnUNetTrainer_warmupnet_100ep_lr1e4_withhead_nosmooth -p nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/multitalent/nnUNet_trained_models/Dataset422_MBH_SEG_pseudo/pretrained/Dataset617_nativect__MultiTalent_trainer_4000ep__nnUNetResEncUNetL1x1x1_Plans_bs24__3d_fullres__fold_all/nnUNetTrainer_warmupnet_1e3_nosmooth__nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT__3d_fullres/fold_2/checkpoint_best.pth"
bsub -gpu num=1:j_exclusive=yes:gmem=20G -q gpu -R tensorcore -o /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs/%J.log -e /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs//%J.err -L /bin/bash "source ~/.bashrc_v2challenges   &&  multitalent_train 402 3d_fullres 3 -tr nnUNetTrainer_warmupnet_100ep_lr1e4_withhead_nosmooth -p nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/multitalent/nnUNet_trained_models/Dataset422_MBH_SEG_pseudo/pretrained/Dataset617_nativect__MultiTalent_trainer_4000ep__nnUNetResEncUNetL1x1x1_Plans_bs24__3d_fullres__fold_all/nnUNetTrainer_warmupnet_1e3_nosmooth__nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT__3d_fullres/fold_3/checkpoint_best.pth"
bsub -gpu num=1:j_exclusive=yes:gmem=20G -q gpu -R tensorcore -o /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs/%J.log -e /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs//%J.err -L /bin/bash "source ~/.bashrc_v2challenges   &&  multitalent_train 402 3d_fullres 4 -tr nnUNetTrainer_warmupnet_100ep_lr1e4_withhead_nosmooth -p nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/multitalent/nnUNet_trained_models/Dataset422_MBH_SEG_pseudo/pretrained/Dataset617_nativect__MultiTalent_trainer_4000ep__nnUNetResEncUNetL1x1x1_Plans_bs24__3d_fullres__fold_all/nnUNetTrainer_warmupnet_1e3_nosmooth__nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT__3d_fullres/fold_4/checkpoint_best.pth"
bsub -gpu num=1:j_exclusive=yes:gmem=20G -q gpu -R tensorcore -o /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs/%J.log -e /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs//%J.err -L /bin/bash "source ~/.bashrc_v2challenges   &&  multitalent_train 402 3d_fullres 0 -tr nnUNetTrainer_warmupnet_100ep_lr1e5_withhead -p nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/multitalent/nnUNet_trained_models/Dataset422_MBH_SEG_pseudo/pretrained/Dataset617_nativect__MultiTalent_trainer_4000ep__nnUNetResEncUNetL1x1x1_Plans_bs24__3d_fullres__fold_all/nnUNetTrainer_warmupnet_1e3_nosmooth__nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT__3d_fullres/fold_0/checkpoint_best.pth"
bsub -gpu num=1:j_exclusive=yes:gmem=20G -q gpu -R tensorcore -o /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs/%J.log -e /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs//%J.err -L /bin/bash "source ~/.bashrc_v2challenges   &&  multitalent_train 402 3d_fullres 1 -tr nnUNetTrainer_warmupnet_100ep_lr1e5_withhead -p nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/multitalent/nnUNet_trained_models/Dataset422_MBH_SEG_pseudo/pretrained/Dataset617_nativect__MultiTalent_trainer_4000ep__nnUNetResEncUNetL1x1x1_Plans_bs24__3d_fullres__fold_all/nnUNetTrainer_warmupnet_1e3_nosmooth__nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT__3d_fullres/fold_1/checkpoint_best.pth"
bsub -gpu num=1:j_exclusive=yes:gmem=20G -q gpu -R tensorcore -o /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs/%J.log -e /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs//%J.err -L /bin/bash "source ~/.bashrc_v2challenges   &&  multitalent_train 402 3d_fullres 2 -tr nnUNetTrainer_warmupnet_100ep_lr1e5_withhead -p nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/multitalent/nnUNet_trained_models/Dataset422_MBH_SEG_pseudo/pretrained/Dataset617_nativect__MultiTalent_trainer_4000ep__nnUNetResEncUNetL1x1x1_Plans_bs24__3d_fullres__fold_all/nnUNetTrainer_warmupnet_1e3_nosmooth__nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT__3d_fullres/fold_2/checkpoint_best.pth"
bsub -gpu num=1:j_exclusive=yes:gmem=20G -q gpu -R tensorcore -o /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs/%J.log -e /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs//%J.err -L /bin/bash "source ~/.bashrc_v2challenges   &&  multitalent_train 402 3d_fullres 3 -tr nnUNetTrainer_warmupnet_100ep_lr1e5_withhead -p nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/multitalent/nnUNet_trained_models/Dataset422_MBH_SEG_pseudo/pretrained/Dataset617_nativect__MultiTalent_trainer_4000ep__nnUNetResEncUNetL1x1x1_Plans_bs24__3d_fullres__fold_all/nnUNetTrainer_warmupnet_1e3_nosmooth__nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT__3d_fullres/fold_3/checkpoint_best.pth"
bsub -gpu num=1:j_exclusive=yes:gmem=20G -q gpu -R tensorcore -o /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs/%J.log -e /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs//%J.err -L /bin/bash "source ~/.bashrc_v2challenges   &&  multitalent_train 402 3d_fullres 4 -tr nnUNetTrainer_warmupnet_100ep_lr1e5_withhead -p nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/multitalent/nnUNet_trained_models/Dataset422_MBH_SEG_pseudo/pretrained/Dataset617_nativect__MultiTalent_trainer_4000ep__nnUNetResEncUNetL1x1x1_Plans_bs24__3d_fullres__fold_all/nnUNetTrainer_warmupnet_1e3_nosmooth__nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT__3d_fullres/fold_4/checkpoint_best.pth"


418
bsub -gpu num=1:j_exclusive=yes:gmem=20G -q gpu -R tensorcore -o /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs/%J.log -e /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs//%J.err -L /bin/bash "source ~/.bashrc_v2challenges   &&  multitalent_train 418 3d_fullres 0 -tr nnUNetTrainer_warmupnet_100ep_lr1e3_withhead_recall -p nnUNetResEncUNetL05x05x05_Plans_znorm_MT -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/multitalent/nnUNet_trained_models/Dataset420_epvs_artificial/pretrained/Dataset619_nativemultistem__MultiTalent_trainer_multistems_4000ep__nnUNetResEncUNetL1x1x1_Plans_znorm_bs24__3d_fullres__fold_all/nnUNetTrainerRecall_warmupnet__nnUNetResEncUNetL05x05x05_Plans_znorm_MT__3d_fullres/fold_0/checkpoint_best.pth"
bsub -gpu num=1:j_exclusive=yes:gmem=20G -q gpu -R tensorcore -o /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs/%J.log -e /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs//%J.err -L /bin/bash "source ~/.bashrc_v2challenges   &&  multitalent_train 418 3d_fullres 1 -tr nnUNetTrainer_warmupnet_100ep_lr1e3_withhead_recall -p nnUNetResEncUNetL05x05x05_Plans_znorm_MT -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/multitalent/nnUNet_trained_models/Dataset420_epvs_artificial/pretrained/Dataset619_nativemultistem__MultiTalent_trainer_multistems_4000ep__nnUNetResEncUNetL1x1x1_Plans_znorm_bs24__3d_fullres__fold_all/nnUNetTrainerRecall_warmupnet__nnUNetResEncUNetL05x05x05_Plans_znorm_MT__3d_fullres/fold_1/checkpoint_best.pth"
bsub -gpu num=1:j_exclusive=yes:gmem=20G -q gpu -R tensorcore -o /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs/%J.log -e /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs//%J.err -L /bin/bash "source ~/.bashrc_v2challenges   &&  multitalent_train 418 3d_fullres 2 -tr nnUNetTrainer_warmupnet_100ep_lr1e3_withhead_recall -p nnUNetResEncUNetL05x05x05_Plans_znorm_MT -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/multitalent/nnUNet_trained_models/Dataset420_epvs_artificial/pretrained/Dataset619_nativemultistem__MultiTalent_trainer_multistems_4000ep__nnUNetResEncUNetL1x1x1_Plans_znorm_bs24__3d_fullres__fold_all/nnUNetTrainerRecall_warmupnet__nnUNetResEncUNetL05x05x05_Plans_znorm_MT__3d_fullres/fold_2/checkpoint_best.pth"
bsub -gpu num=1:j_exclusive=yes:gmem=20G -q gpu -R tensorcore -o /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs/%J.log -e /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs//%J.err -L /bin/bash "source ~/.bashrc_v2challenges   &&  multitalent_train 418 3d_fullres 3 -tr nnUNetTrainer_warmupnet_100ep_lr1e3_withhead_recall -p nnUNetResEncUNetL05x05x05_Plans_znorm_MT -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/multitalent/nnUNet_trained_models/Dataset420_epvs_artificial/pretrained/Dataset619_nativemultistem__MultiTalent_trainer_multistems_4000ep__nnUNetResEncUNetL1x1x1_Plans_znorm_bs24__3d_fullres__fold_all/nnUNetTrainerRecall_warmupnet__nnUNetResEncUNetL05x05x05_Plans_znorm_MT__3d_fullres/fold_3/checkpoint_best.pth"
bsub -gpu num=1:j_exclusive=yes:gmem=20G -q gpu -R tensorcore -o /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs/%J.log -e /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs//%J.err -L /bin/bash "source ~/.bashrc_v2challenges   &&  multitalent_train 418 3d_fullres 4 -tr nnUNetTrainer_warmupnet_100ep_lr1e3_withhead_recall -p nnUNetResEncUNetL05x05x05_Plans_znorm_MT -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/multitalent/nnUNet_trained_models/Dataset420_epvs_artificial/pretrained/Dataset619_nativemultistem__MultiTalent_trainer_multistems_4000ep__nnUNetResEncUNetL1x1x1_Plans_znorm_bs24__3d_fullres__fold_all/nnUNetTrainerRecall_warmupnet__nnUNetResEncUNetL05x05x05_Plans_znorm_MT__3d_fullres/fold_4/checkpoint_best.pth"
bsub -gpu num=1:j_exclusive=yes:gmem=20G -q gpu -R tensorcore -o /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs/%J.log -e /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs//%J.err -L /bin/bash "source ~/.bashrc_v2challenges   &&  multitalent_train 418 3d_fullres 0 -tr nnUNetTrainer_warmupnet_100ep_lr1e4_withhead_recall -p nnUNetResEncUNetL05x05x05_Plans_znorm_MT -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/multitalent/nnUNet_trained_models/Dataset420_epvs_artificial/pretrained/Dataset619_nativemultistem__MultiTalent_trainer_multistems_4000ep__nnUNetResEncUNetL1x1x1_Plans_znorm_bs24__3d_fullres__fold_all/nnUNetTrainerRecall_warmupnet__nnUNetResEncUNetL05x05x05_Plans_znorm_MT__3d_fullres/fold_0/checkpoint_best.pth"
bsub -gpu num=1:j_exclusive=yes:gmem=20G -q gpu -R tensorcore -o /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs/%J.log -e /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs//%J.err -L /bin/bash "source ~/.bashrc_v2challenges   &&  multitalent_train 418 3d_fullres 1 -tr nnUNetTrainer_warmupnet_100ep_lr1e4_withhead_recall -p nnUNetResEncUNetL05x05x05_Plans_znorm_MT -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/multitalent/nnUNet_trained_models/Dataset420_epvs_artificial/pretrained/Dataset619_nativemultistem__MultiTalent_trainer_multistems_4000ep__nnUNetResEncUNetL1x1x1_Plans_znorm_bs24__3d_fullres__fold_all/nnUNetTrainerRecall_warmupnet__nnUNetResEncUNetL05x05x05_Plans_znorm_MT__3d_fullres/fold_1/checkpoint_best.pth"
bsub -gpu num=1:j_exclusive=yes:gmem=20G -q gpu -R tensorcore -o /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs/%J.log -e /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs//%J.err -L /bin/bash "source ~/.bashrc_v2challenges   &&  multitalent_train 418 3d_fullres 2 -tr nnUNetTrainer_warmupnet_100ep_lr1e4_withhead_recall -p nnUNetResEncUNetL05x05x05_Plans_znorm_MT -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/multitalent/nnUNet_trained_models/Dataset420_epvs_artificial/pretrained/Dataset619_nativemultistem__MultiTalent_trainer_multistems_4000ep__nnUNetResEncUNetL1x1x1_Plans_znorm_bs24__3d_fullres__fold_all/nnUNetTrainerRecall_warmupnet__nnUNetResEncUNetL05x05x05_Plans_znorm_MT__3d_fullres/fold_2/checkpoint_best.pth"
bsub -gpu num=1:j_exclusive=yes:gmem=20G -q gpu -R tensorcore -o /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs/%J.log -e /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs//%J.err -L /bin/bash "source ~/.bashrc_v2challenges   &&  multitalent_train 418 3d_fullres 3 -tr nnUNetTrainer_warmupnet_100ep_lr1e4_withhead_recall -p nnUNetResEncUNetL05x05x05_Plans_znorm_MT -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/multitalent/nnUNet_trained_models/Dataset420_epvs_artificial/pretrained/Dataset619_nativemultistem__MultiTalent_trainer_multistems_4000ep__nnUNetResEncUNetL1x1x1_Plans_znorm_bs24__3d_fullres__fold_all/nnUNetTrainerRecall_warmupnet__nnUNetResEncUNetL05x05x05_Plans_znorm_MT__3d_fullres/fold_3/checkpoint_best.pth"
bsub -gpu num=1:j_exclusive=yes:gmem=20G -q gpu -R tensorcore -o /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs/%J.log -e /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs//%J.err -L /bin/bash "source ~/.bashrc_v2challenges   &&  multitalent_train 418 3d_fullres 4 -tr nnUNetTrainer_warmupnet_100ep_lr1e4_withhead_recall -p nnUNetResEncUNetL05x05x05_Plans_znorm_MT -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/multitalent/nnUNet_trained_models/Dataset420_epvs_artificial/pretrained/Dataset619_nativemultistem__MultiTalent_trainer_multistems_4000ep__nnUNetResEncUNetL1x1x1_Plans_znorm_bs24__3d_fullres__fold_all/nnUNetTrainerRecall_warmupnet__nnUNetResEncUNetL05x05x05_Plans_znorm_MT__3d_fullres/fold_4/checkpoint_best.pth"
bsub -gpu num=1:j_exclusive=yes:gmem=20G -q gpu -R tensorcore -o /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs/%J.log -e /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs//%J.err -L /bin/bash "source ~/.bashrc_v2challenges   &&  multitalent_train 418 3d_fullres 0 -tr nnUNetTrainer_warmupnet_100ep_lr1e5_withhead_recall -p nnUNetResEncUNetL05x05x05_Plans_znorm_MT -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/multitalent/nnUNet_trained_models/Dataset420_epvs_artificial/pretrained/Dataset619_nativemultistem__MultiTalent_trainer_multistems_4000ep__nnUNetResEncUNetL1x1x1_Plans_znorm_bs24__3d_fullres__fold_all/nnUNetTrainerRecall_warmupnet__nnUNetResEncUNetL05x05x05_Plans_znorm_MT__3d_fullres/fold_0/checkpoint_best.pth"
bsub -gpu num=1:j_exclusive=yes:gmem=20G -q gpu -R tensorcore -o /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs/%J.log -e /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs//%J.err -L /bin/bash "source ~/.bashrc_v2challenges   &&  multitalent_train 418 3d_fullres 1 -tr nnUNetTrainer_warmupnet_100ep_lr1e5_withhead_recall -p nnUNetResEncUNetL05x05x05_Plans_znorm_MT -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/multitalent/nnUNet_trained_models/Dataset420_epvs_artificial/pretrained/Dataset619_nativemultistem__MultiTalent_trainer_multistems_4000ep__nnUNetResEncUNetL1x1x1_Plans_znorm_bs24__3d_fullres__fold_all/nnUNetTrainerRecall_warmupnet__nnUNetResEncUNetL05x05x05_Plans_znorm_MT__3d_fullres/fold_1/checkpoint_best.pth"
bsub -gpu num=1:j_exclusive=yes:gmem=20G -q gpu -R tensorcore -o /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs/%J.log -e /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs//%J.err -L /bin/bash "source ~/.bashrc_v2challenges   &&  multitalent_train 418 3d_fullres 2 -tr nnUNetTrainer_warmupnet_100ep_lr1e5_withhead_recall -p nnUNetResEncUNetL05x05x05_Plans_znorm_MT -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/multitalent/nnUNet_trained_models/Dataset420_epvs_artificial/pretrained/Dataset619_nativemultistem__MultiTalent_trainer_multistems_4000ep__nnUNetResEncUNetL1x1x1_Plans_znorm_bs24__3d_fullres__fold_all/nnUNetTrainerRecall_warmupnet__nnUNetResEncUNetL05x05x05_Plans_znorm_MT__3d_fullres/fold_2/checkpoint_best.pth"
bsub -gpu num=1:j_exclusive=yes:gmem=20G -q gpu -R tensorcore -o /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs/%J.log -e /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs//%J.err -L /bin/bash "source ~/.bashrc_v2challenges   &&  multitalent_train 418 3d_fullres 3 -tr nnUNetTrainer_warmupnet_100ep_lr1e5_withhead_recall -p nnUNetResEncUNetL05x05x05_Plans_znorm_MT -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/multitalent/nnUNet_trained_models/Dataset420_epvs_artificial/pretrained/Dataset619_nativemultistem__MultiTalent_trainer_multistems_4000ep__nnUNetResEncUNetL1x1x1_Plans_znorm_bs24__3d_fullres__fold_all/nnUNetTrainerRecall_warmupnet__nnUNetResEncUNetL05x05x05_Plans_znorm_MT__3d_fullres/fold_3/checkpoint_best.pth"
bsub -gpu num=1:j_exclusive=yes:gmem=20G -q gpu -R tensorcore -o /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs/%J.log -e /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs//%J.err -L /bin/bash "source ~/.bashrc_v2challenges   &&  multitalent_train 418 3d_fullres 4 -tr nnUNetTrainer_warmupnet_100ep_lr1e5_withhead_recall -p nnUNetResEncUNetL05x05x05_Plans_znorm_MT -pretrained_weights /dkfz/cluster/gpu/checkpoints/OE0441/c306h/multitalent/nnUNet_trained_models/Dataset420_epvs_artificial/pretrained/Dataset619_nativemultistem__MultiTalent_trainer_multistems_4000ep__nnUNetResEncUNetL1x1x1_Plans_znorm_bs24__3d_fullres__fold_all/nnUNetTrainerRecall_warmupnet__nnUNetResEncUNetL05x05x05_Plans_znorm_MT__3d_fullres/fold_4/checkpoint_best.pth"


predictions MBH-SEG

bsub -gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=33G -q gpu -R  "select[hname!='e230-dgx2-2']" -R  "select[hname!='e071-gpu05']"  -R  "select[hname!='e230-dgx1-1']"  -R tensorcore   -R  "select[hname!='lsf22-gpu06']"   -o /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs/%J.log -e /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs//%J.err -L /bin/bash "source ~/.bashrc_v2native  && multitalent_predict_from_modelfolder  -i /omics/groups/OE0441/E132-Projekte/Projects/2024_ulrich_mbhseg/FirstStage_test/ -o /omics/groups/OE0441/E132-Projekte/Projects/2024_ulrich_mbhseg/nnUNetTrainer_warmupnet_100ep_lr1e4_withhead__nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT__3d_fullres -m /dkfz/cluster/gpu/checkpoints/OE0441/c306h//multitalent/nnUNet_trained_models/Dataset402_MBH_SEG/pretrained/unknown_pretraining/nnUNetTrainer_warmupnet_100ep_lr1e4_withhead__nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT__3d_fullres --save_probabilities -chk checkpoint_best.pth"
bsub -gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=33G -q gpu -R  "select[hname!='e230-dgx2-2']" -R  "select[hname!='e071-gpu05']"  -R  "select[hname!='e230-dgx1-1']"  -R tensorcore   -R  "select[hname!='lsf22-gpu06']"   -o /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs/%J.log -e /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs//%J.err -L /bin/bash "source ~/.bashrc_v2native  && multitalent_predict_from_modelfolder  -i /omics/groups/OE0441/E132-Projekte/Projects/2024_ulrich_mbhseg/FirstStage_test/ -o /omics/groups/OE0441/E132-Projekte/Projects/2024_ulrich_mbhseg/nnUNetTrainer_warmupnet_100ep_lr1e3_withhead__nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT__3d_fullres -m /dkfz/cluster/gpu/checkpoints/OE0441/c306h//multitalent/nnUNet_trained_models/Dataset402_MBH_SEG/pretrained/unknown_pretraining/nnUNetTrainer_warmupnet_100ep_lr1e3_withhead__nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT__3d_fullres --save_probabilities -chk checkpoint_best.pth"
bsub -gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=10G -q gpu -R  "select[hname!='e230-dgx2-2']" -R  "select[hname!='e071-gpu05']"  -R  "select[hname!='e230-dgx1-1']"  -R tensorcore   -R  "select[hname!='lsf22-gpu06']"   -o /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs/%J.log -e /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs//%J.err -L /bin/bash "source ~/.bashrc_v2native  && multitalent_predict_from_modelfolder  -i /omics/groups/OE0441/E132-Projekte/Projects/2024_ulrich_mbhseg/FirstStage_test/ -o /omics/groups/OE0441/E132-Projekte/Projects/2024_ulrich_mbhseg/nnUNetTrainer_warmupnet_100ep_lr1e4_nosmooth__nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT__3d_fullres -m /dkfz/cluster/gpu/checkpoints/OE0441/c306h//multitalent/nnUNet_trained_models/Dataset402_MBH_SEG/pretrained/unknown_pretraining/nnUNetTrainer_warmupnet_100ep_lr1e4_nosmooth__nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT__3d_fullres --save_probabilities -chk checkpoint_best.pth"
bsub -gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=33G -q gpu -R  "select[hname!='e230-dgx2-2']" -R  "select[hname!='e071-gpu05']"  -R  "select[hname!='e230-dgx1-1']"  -R tensorcore   -R  "select[hname!='lsf22-gpu06']"   -o /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs/%J.log -e /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs//%J.err -L /bin/bash "source ~/.bashrc_v2native  && multitalent_predict_from_modelfolder  -i /omics/groups/OE0441/E132-Projekte/Projects/2024_ulrich_mbhseg/FirstStage_test/ -o /omics/groups/OE0441/E132-Projekte/Projects/2024_ulrich_mbhseg/nnUNetTrainer_warmupnet_100ep_lr1e3_withhead_nosmooth__nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT__3d_fullres -m /dkfz/cluster/gpu/checkpoints/OE0441/c306h//multitalent/nnUNet_trained_models/Dataset402_MBH_SEG/pretrained/unknown_pretraining/nnUNetTrainer_warmupnet_100ep_lr1e3_withhead_nosmooth__nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT__3d_fullres --save_probabilities -chk checkpoint_best.pth"
bsub -gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=33G -q gpu -R  "select[hname!='e230-dgx2-2']" -R  "select[hname!='e071-gpu05']"  -R  "select[hname!='e230-dgx1-1']"  -R tensorcore   -R  "select[hname!='lsf22-gpu06']"   -o /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs/%J.log -e /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs//%J.err -L /bin/bash "source ~/.bashrc_v2native  && multitalent_predict_from_modelfolder  -i /omics/groups/OE0441/E132-Projekte/Projects/2024_ulrich_mbhseg/FirstStage_test/ -o /omics/groups/OE0441/E132-Projekte/Projects/2024_ulrich_mbhseg/nnUNetTrainer_warmupnet_100ep_lr1e4_withhead_nosmooth__nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT__3d_fullres -m /dkfz/cluster/gpu/checkpoints/OE0441/c306h//multitalent/nnUNet_trained_models/Dataset402_MBH_SEG/pretrained/unknown_pretraining/nnUNetTrainer_warmupnet_100ep_lr1e4_withhead_nosmooth__nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT__3d_fullres --save_probabilities -chk checkpoint_best.pth"

bsub -gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=10G -q gpu -R  "select[hname!='e230-dgx2-2']" -R  "select[hname!='e071-gpu05']"  -R  "select[hname!='e230-dgx1-1']"  -R tensorcore   -R  "select[hname!='lsf22-gpu06']"   -o /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs/%J.log -e /dkfz/cluster/gpu/checkpoints/OE0441/c306h/logs//%J.err -L /bin/bash "source ~/.bashrc_v2native  && multitalent_ensemble -i /omics/groups/OE0441/E132-Projekte/Projects/2024_ulrich_mbhseg/nnUNetTrainer_warmupnet_100ep_lr1e4_withhead_nosmooth__nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT__3d_fullres /omics/groups/OE0441/E132-Projekte/Projects/2024_ulrich_mbhseg/nnUNetTrainer_warmupnet_100ep_lr1e3_withhead_nosmooth__nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT__3d_fullres /omics/groups/OE0441/E132-Projekte/Projects/2024_ulrich_mbhseg/nnUNetTrainer_warmupnet_100ep_lr1e4_nosmooth__nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT__3d_fullres /omics/groups/OE0441/E132-Projekte/Projects/2024_ulrich_mbhseg/nnUNetTrainer_warmupnet_100ep_lr1e3_withhead__nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT__3d_fullres /omics/groups/OE0441/E132-Projekte/Projects/2024_ulrich_mbhseg/nnUNetTrainer_warmupnet_100ep_lr1e4_withhead__nnUNetResEncUNetL1x1x1_Plans_orgnorm_MT__3d_fullres -o /omics/groups/OE0441/E132-Projekte/Projects/2024_ulrich_mbhseg/FirstStage_test/ensemble
'''