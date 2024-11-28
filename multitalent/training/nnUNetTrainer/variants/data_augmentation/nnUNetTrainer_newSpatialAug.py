from typing import Union, Tuple, List

import numpy as np
from batchgenerators.transforms.abstract_transforms import AbstractTransform
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor

from multitalent.training.data_augmentation.custom_transforms.cascade_transforms import MoveSegAsOneHotToData, \
    ApplyRandomBinaryOperatorTransform, RemoveRandomConnectedComponentFromOneHotEncodingTransform
from multitalent.training.data_augmentation.custom_transforms.cleaner_spatial_transform import CleanerSpatialTransform
from multitalent.training.data_augmentation.custom_transforms.deep_supervision_donwsampling import \
    DownsampleSegForDSTransform2
from multitalent.training.data_augmentation.custom_transforms.masking import MaskTransform
from multitalent.training.data_augmentation.custom_transforms.region_based_training import \
    ConvertSegmentationToRegionsTransform
from multitalent.training.data_augmentation.custom_transforms.transforms_for_dummy_2d import Convert2DTo3DTransform, \
    Convert3DTo2DTransform
from multitalent.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_newSpatialAug(nnUNetTrainer):
    @staticmethod
    def get_training_transforms(patch_size: Union[np.ndarray, Tuple[int]],
                                rotation_for_DA: dict,
                                deep_supervision_scales: Union[List, Tuple],
                                mirror_axes: Tuple[int, ...],
                                do_dummy_2d_data_aug: bool,
                                order_resampling_data: int = 3,
                                order_resampling_seg: int = 1,
                                border_val_seg: int = -1,
                                use_mask_for_norm: List[bool] = None,
                                is_cascaded: bool = False,
                                foreground_labels: Union[Tuple[int, ...], List[int]] = None,
                                regions: List[Union[List[int], Tuple[int, ...], int]] = None,
                                ignore_label: int = None) -> AbstractTransform:
        tr_transforms = []
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            tr_transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None

        tr_transforms.append(CleanerSpatialTransform(tuple(patch_size_spatial), patch_center_dist_from_border=0,
                                                     p_el_per_sample=0,
                                                     p_rot_per_sample=0.2, angle_x=rotation_for_DA['x'],
                                                     angle_y=rotation_for_DA['y'], angle_z=rotation_for_DA['z'],
                                                     p_rot_per_axis=1, p_scale_per_sample=0.2, scale=(0.7, 1.4),
                                                     p_independent_scale_per_axis=0, border_mode_data='constant',
                                                     border_cval_data=0, interpolation_order_data=order_resampling_data,
                                                     border_mode_seg='constant', border_cval_seg=border_val_seg,
                                                     interpolation_order_seg=order_resampling_seg, random_crop=False
                                                     ))

        if do_dummy_2d_data_aug:
            tr_transforms.append(Convert2DTo3DTransform())

        tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
        tr_transforms.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
                                                   p_per_channel=0.5))
        tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))
        tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
        tr_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                                            p_per_channel=0.5,
                                                            order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                                            ignore_axes=ignore_axes))
        tr_transforms.append(GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1))
        tr_transforms.append(GammaTransform((0.7, 1.5), False, True, retain_stats=True, p_per_sample=0.3))

        if mirror_axes is not None and len(mirror_axes) > 0:
            tr_transforms.append(MirrorTransform(mirror_axes))

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            tr_transforms.append(MaskTransform([i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                                               mask_idx_in_seg=0, set_outside_to=0))

        tr_transforms.append(RemoveLabelTransform(-1, 0))

        if is_cascaded:
            assert foreground_labels is not None, 'We need foreground_labels for cascade augmentations'
            tr_transforms.append(MoveSegAsOneHotToData(1, foreground_labels, 'seg', 'data'))
            tr_transforms.append(ApplyRandomBinaryOperatorTransform(
                channel_idx=list(range(-len(foreground_labels), 0)),
                p_per_sample=0.4,
                key="data",
                strel_size=(1, 8),
                p_per_label=1))
            tr_transforms.append(
                RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                    channel_idx=list(range(-len(foreground_labels), 0)),
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


class nnUNetTrainer_newSpatialAug_withElDef(nnUNetTrainer):
    @staticmethod
    def get_training_transforms(patch_size: Union[np.ndarray, Tuple[int]],
                                rotation_for_DA: dict,
                                deep_supervision_scales: Union[List, Tuple],
                                mirror_axes: Tuple[int, ...],
                                do_dummy_2d_data_aug: bool,
                                order_resampling_data: int = 3,
                                order_resampling_seg: int = 1,
                                border_val_seg: int = -1,
                                use_mask_for_norm: List[bool] = None,
                                is_cascaded: bool = False,
                                foreground_labels: Union[Tuple[int, ...], List[int]] = None,
                                regions: List[Union[List[int], Tuple[int, ...], int]] = None,
                                ignore_label: int = None) -> AbstractTransform:
        if do_dummy_2d_data_aug:
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size

        transforms = nnUNetTrainer_newSpatialAug.get_training_transforms(patch_size, rotation_for_DA,
                                                                         deep_supervision_scales, mirror_axes,
                                                                         do_dummy_2d_data_aug, order_resampling_data,
                                                                         order_resampling_seg, border_val_seg,
                                                                         use_mask_for_norm, is_cascaded,
                                                                         foreground_labels, regions, ignore_label)
        idx = np.where([isinstance(i, (SpatialTransform, CleanerSpatialTransform)) for i in transforms.transforms])[0][0]
        transforms.transforms[idx] = CleanerSpatialTransform(tuple(patch_size_spatial), patch_center_dist_from_border=0,
                                                           p_el_per_sample=0.2,
                                                           deformation_scale=lambda shape, dim: np.random.uniform(0,
                                                                                                                  max(shape) / 4),
                                                           deformation_magnitude=lambda shape, dim,
                                                                                        scale: np.random.uniform(
                                                               max(-scale, -max(shape) / 4),
                                                               min(scale, max(shape) / 4)),
                                                           p_rot_per_sample=0.2, angle_x=rotation_for_DA['x'],
                                                           angle_y=rotation_for_DA['y'], angle_z=rotation_for_DA['z'],
                                                           p_rot_per_axis=1, p_scale_per_sample=0.2, scale=(0.7, 1.4),
                                                           p_independent_scale_per_axis=0, border_mode_data='constant',
                                                           border_cval_data=0,
                                                           interpolation_order_data=order_resampling_data,
                                                           border_mode_seg='constant', border_cval_seg=border_val_seg,
                                                           interpolation_order_seg=order_resampling_seg,
                                                           random_crop=False
                                                           )
        return transforms

class nnUNetTrainer_newSpatialAug_withElDef_noPref(nnUNetTrainer_newSpatialAug_withElDef):
    @staticmethod
    def get_training_transforms(patch_size: Union[np.ndarray, Tuple[int]],
                                rotation_for_DA: dict,
                                deep_supervision_scales: Union[List, Tuple],
                                mirror_axes: Tuple[int, ...],
                                do_dummy_2d_data_aug: bool,
                                order_resampling_data: int = 3,
                                order_resampling_seg: int = 1,
                                border_val_seg: int = -1,
                                use_mask_for_norm: List[bool] = None,
                                is_cascaded: bool = False,
                                foreground_labels: Union[Tuple[int, ...], List[int]] = None,
                                regions: List[Union[List[int], Tuple[int, ...], int]] = None,
                                ignore_label: int = None) -> AbstractTransform:

        transforms = nnUNetTrainer_newSpatialAug_withElDef.get_training_transforms(patch_size, rotation_for_DA,
                                                                         deep_supervision_scales, mirror_axes,
                                                                         do_dummy_2d_data_aug, order_resampling_data,
                                                                         order_resampling_seg, border_val_seg,
                                                                         use_mask_for_norm, is_cascaded,
                                                                         foreground_labels, regions, ignore_label)
        idx = np.where([isinstance(i, (SpatialTransform, CleanerSpatialTransform)) for i in transforms.transforms])[0][0]
        transforms.transforms[idx].prefilter = False
        return transforms


class nnUNetTrainer_newSpatialAug_noPref(nnUNetTrainer):
    @staticmethod
    def get_training_transforms(patch_size: Union[np.ndarray, Tuple[int]],
                                rotation_for_DA: dict,
                                deep_supervision_scales: Union[List, Tuple],
                                mirror_axes: Tuple[int, ...],
                                do_dummy_2d_data_aug: bool,
                                order_resampling_data: int = 3,
                                order_resampling_seg: int = 1,
                                border_val_seg: int = -1,
                                use_mask_for_norm: List[bool] = None,
                                is_cascaded: bool = False,
                                foreground_labels: Union[Tuple[int, ...], List[int]] = None,
                                regions: List[Union[List[int], Tuple[int, ...], int]] = None,
                                ignore_label: int = None) -> AbstractTransform:

        transforms = nnUNetTrainer_newSpatialAug.get_training_transforms(patch_size, rotation_for_DA,
                                                                         deep_supervision_scales, mirror_axes,
                                                                         do_dummy_2d_data_aug, order_resampling_data,
                                                                         order_resampling_seg, border_val_seg,
                                                                         use_mask_for_norm, is_cascaded,
                                                                         foreground_labels, regions, ignore_label)
        idx = np.where([isinstance(i, (SpatialTransform, CleanerSpatialTransform)) for i in transforms.transforms])[0][0]
        transforms.transforms[idx].prefilter = False
        return transforms



class nnUNetTrainer_newSpatialAug_withElDef2(nnUNetTrainer):
    @staticmethod
    def get_training_transforms(patch_size: Union[np.ndarray, Tuple[int]],
                                rotation_for_DA: dict,
                                deep_supervision_scales: Union[List, Tuple],
                                mirror_axes: Tuple[int, ...],
                                do_dummy_2d_data_aug: bool,
                                order_resampling_data: int = 3,
                                order_resampling_seg: int = 1,
                                border_val_seg: int = -1,
                                use_mask_for_norm: List[bool] = None,
                                is_cascaded: bool = False,
                                foreground_labels: Union[Tuple[int, ...], List[int]] = None,
                                regions: List[Union[List[int], Tuple[int, ...], int]] = None,
                                ignore_label: int = None) -> AbstractTransform:
        if do_dummy_2d_data_aug:
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size

        transforms = nnUNetTrainer_newSpatialAug.get_training_transforms(patch_size, rotation_for_DA,
                                                                         deep_supervision_scales, mirror_axes,
                                                                         do_dummy_2d_data_aug, order_resampling_data,
                                                                         order_resampling_seg, border_val_seg,
                                                                         use_mask_for_norm, is_cascaded,
                                                                         foreground_labels, regions, ignore_label)
        idx = np.where([isinstance(i, (SpatialTransform, CleanerSpatialTransform)) for i in transforms.transforms])[0][0]
        transforms.transforms[idx] = CleanerSpatialTransform(tuple(patch_size_spatial), patch_center_dist_from_border=0,
                                                           p_el_per_sample=0.2,
                                                           deformation_scale=lambda shape, dim: np.random.uniform(0, shape[dim] / 5),
                                                           deformation_magnitude=lambda shape, dim, scale: np.random.uniform(scale / 3, min(scale * 0.9, max(shape) / 4)),
                                                           p_rot_per_sample=0.2, angle_x=rotation_for_DA['x'],
                                                           angle_y=rotation_for_DA['y'], angle_z=rotation_for_DA['z'],
                                                           p_rot_per_axis=1, p_scale_per_sample=0.2, scale=(0.7, 1.4),
                                                           p_independent_scale_per_axis=0, border_mode_data='constant',
                                                           border_cval_data=0,
                                                           interpolation_order_data=order_resampling_data,
                                                           border_mode_seg='constant', border_cval_seg=border_val_seg,
                                                           interpolation_order_seg=order_resampling_seg,
                                                           random_crop=False
                                                           )
        return transforms


class nnUNetTrainer_newSpatialAug_withElDef3(nnUNetTrainer):
    @staticmethod
    def get_training_transforms(patch_size: Union[np.ndarray, Tuple[int]],
                                rotation_for_DA: dict,
                                deep_supervision_scales: Union[List, Tuple],
                                mirror_axes: Tuple[int, ...],
                                do_dummy_2d_data_aug: bool,
                                order_resampling_data: int = 3,
                                order_resampling_seg: int = 1,
                                border_val_seg: int = -1,
                                use_mask_for_norm: List[bool] = None,
                                is_cascaded: bool = False,
                                foreground_labels: Union[Tuple[int, ...], List[int]] = None,
                                regions: List[Union[List[int], Tuple[int, ...], int]] = None,
                                ignore_label: int = None) -> AbstractTransform:
        if do_dummy_2d_data_aug:
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size

        transforms = nnUNetTrainer_newSpatialAug.get_training_transforms(patch_size, rotation_for_DA,
                                                                         deep_supervision_scales, mirror_axes,
                                                                         do_dummy_2d_data_aug, order_resampling_data,
                                                                         order_resampling_seg, border_val_seg,
                                                                         use_mask_for_norm, is_cascaded,
                                                                         foreground_labels, regions, ignore_label)
        idx = np.where([isinstance(i, (SpatialTransform, CleanerSpatialTransform)) for i in transforms.transforms])[0][0]
        transforms.transforms[idx] = CleanerSpatialTransform(tuple(patch_size_spatial), patch_center_dist_from_border=0,
                                                           p_el_per_sample=0.2,
                                                           deformation_scale=lambda shape, dim: np.random.uniform(0, shape[dim] / 5),
                                                           deformation_magnitude=lambda shape, dim, scale: np.random.uniform(scale / 3, min(scale * 0.9, max(shape) / 4)),
                                                           p_rot_per_sample=0.2, angle_x=rotation_for_DA['x'],
                                                           angle_y=rotation_for_DA['y'], angle_z=rotation_for_DA['z'],
                                                           p_rot_per_axis=0.9, p_scale_per_sample=0.2, scale=(0.7, 1.4),
                                                           p_independent_scale_per_axis=0.5, border_mode_data='constant',
                                                           border_cval_data=0,
                                                           interpolation_order_data=order_resampling_data,
                                                           border_mode_seg='constant', border_cval_seg=border_val_seg,
                                                           interpolation_order_seg=order_resampling_seg,
                                                           random_crop=False
                                                           )
        return transforms
