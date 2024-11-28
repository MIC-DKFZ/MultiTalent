import gc
from copy import deepcopy
from typing import Tuple, Union, Optional

import pandas as pd
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.augmentations.utils import rotate_coords_3d, rotate_coords_2d
from batchgenerators.transforms.abstract_transforms import AbstractTransform
import numpy as np
from numpy import ScalarType as NpScalarType
from batchgenerators.utilities.custom_types import ScalarType, sample_scalar
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice
from scipy.ndimage import gaussian_filter, fourier_gaussian, map_coordinates


def crop(*arrays,
         crop_size: Union[int, Tuple[int, ...]] = 128,
         margins: Union[int, Tuple[int, ...]] = (0, 0, 0),
         crop_type: str = "center",
         pad_modes: Union[str, Tuple[str, ...]] = 'constant',
         pad_kwargs: Union[dict, Tuple[dict, ...]] = {'constant_values': 0}
         ):
    """
    arrays can be nD, BUT the first axis is interpreted as color channel and will NOT be cropped!
    """
    shapes = [i.shape[1:] for i in arrays if i is not None]
    assert all([i == shapes[0] for i in shapes[1:]]), "All shapes must be the same!"
    shape = shapes[0]

    if isinstance(crop_size, NpScalarType):
        crop_size = tuple([crop_size] * len(shape))
    else:
        assert len(crop_size) == len(shape)
        assert isinstance(crop_size, tuple)

    if isinstance(margins, NpScalarType):
        margins = tuple([margins] * len(shape))
    else:
        assert len(margins) == len(shape)
        assert isinstance(margins, tuple)

    if isinstance(pad_modes, str):
        pad_modes = tuple([pad_modes] * len(arrays))

    if isinstance(pad_kwargs, dict):
        pad_kwargs = tuple([pad_kwargs] * len(arrays))

    if crop_type == 'center':
        bbox = [(max(0, s // 2 - cs // 2), min(s, s // 2 - cs // 2 + cs)) for s, cs in zip(shape, crop_size)]
        slicer = bounding_box_to_slice(bbox)
    else:
        # 2 situations
        # - an axis is smaller than crop_size, don't crop that axis and later pad it with pad_nd_image
        # OR an axis is larger than crop_size but does not allow for the margin => Center crop this axis
        # - an axis is large enough. Do random crop
        bbox = []
        for s, cs, m in zip(shape, crop_size, margins):
            if s < (cs + 2 * m):
                bbox.append(
                    (max(0, s // 2 - cs // 2), min(s, s // 2 - cs // 2 + cs))
                )
            else:
                lb = np.random.randint(low=m, high=s - m - cs + 1)  # + 1 because high is not inclusive!
                bbox.append((lb, lb + cs))
        slicer = bounding_box_to_slice(bbox)

    ret = [a[(slice(0, a.shape[0]), *slicer)] if a is not None else None for a in arrays]
    if ret[0].shape[1:] != crop_size:
        # this happens if cop_size > array_size in at least one dimension.
        ret = [pad_nd_image(i, crop_size, pm, pkwargs) if
               i is not None else None for i, pm, pkwargs in zip(ret, pad_modes, pad_kwargs)]
    return ret


class CleanerSpatialTransform(AbstractTransform):
    def __init__(self, patch_size: Tuple[int, ...],
                 patch_center_dist_from_border: Union[int, Tuple[int, ...]] = 30,
                 p_el_per_sample: float = 1,
                 deformation_scale: ScalarType = lambda shape, dim: np.random.uniform(0, max(shape) / 4),
                 deformation_magnitude: ScalarType = lambda shape, dim, scale: np.random.uniform(max(-scale, -max(shape) / 4),
                                                                                          min(scale, max(shape) / 4)),
                 p_rot_per_sample: float = 1,
                 angle_x: ScalarType = (0, 2 * np.pi),
                 angle_y: ScalarType = (0, 2 * np.pi),
                 angle_z: ScalarType = (0, 2 * np.pi),
                 p_rot_per_axis: float = 1,
                 p_scale_per_sample: float = 1,
                 scale: ScalarType = (0.75, 1.25),
                 p_independent_scale_per_axis: float = 1,
                 border_mode_data: str = 'nearest', border_cval_data: float = 0, interpolation_order_data: int = 3,
                 border_mode_seg: str = 'constant', border_cval_seg: float = 0, interpolation_order_seg: int = 0,
                 random_crop: bool = True,
                 data_key: str = "data", seg_key: str = "seg", regr_target_key: str = None
                 ):
        """
        all ScalarTypes (except deformation_magnitude) need to be function(patch_shape, current_dimension) -> float if they are callable!
        scale needs to accept None for current_dimension if all dimensions are to be scaled simultaneously
        deformation_magnitude nees to be function(patch_shape, current_dimension, deformation_scale) -> float, where deformation_scale is the current scale
        """
        super().__init__()

        # elastic deformation
        self.p_el_per_sample = p_el_per_sample
        self.deformation_scale = deformation_scale
        self.deformation_magnitude = deformation_magnitude

        # scaling
        self.p_scale_per_sample = p_scale_per_sample
        self.scale = scale
        self.p_independent_scale_per_axis = p_independent_scale_per_axis

        # rotation
        self.p_rot_per_sample = p_rot_per_sample
        self.angle_x = angle_x
        self.angle_y = angle_y
        self.angle_z = angle_z
        self.p_rot_per_axis = p_rot_per_axis

        # resampling/interpolation
        self.border_mode_data = border_mode_data
        self.border_cval_data = border_cval_data
        self.order_data = interpolation_order_data

        self.border_mode_seg = border_mode_seg
        self.border_cval_seg = border_cval_seg
        self.order_seg = interpolation_order_seg

        self.border_mode_regr_target = self.border_mode_data
        self.border_cval_regr_target = self.border_cval_data
        self.order_regr_target = self.order_data

        # cropping
        self.random_crop = random_crop
        self.patch_size = patch_size
        self.patch_center_dist_from_border = patch_center_dist_from_border if not isinstance(
            patch_center_dist_from_border, NpScalarType) else tuple([patch_center_dist_from_border] * len(patch_size))

        # dictionary keys
        self.data_key = data_key
        self.seg_key = seg_key
        self.regr_target_key = regr_target_key

        self.return_data, self.return_seg, self.return_regr_target = None, None, None  # saves memory

        self.prefilter = True

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.seg_key)
        regr_target = data_dict.get(self.regr_target_key)

        assert data is not None and 4 <= len(
            data.shape) <= 5, "data_key must exist and data can only be 2D or 3D data (4d or 5d data value)"

        self.return_data = np.zeros_like(data, shape=(*data.shape[:2], *self.patch_size))
        self.return_seg = np.zeros_like(seg, shape=(*seg.shape[:2], *self.patch_size)) if seg is not None else None
        self.return_regr_target = np.zeros_like(regr_target, shape=(
            *regr_target.shape[:2], *self.patch_size)) if regr_target is not None else None

        for sample_idx in range(data.shape[0]):
            # augment_spatial writes to self.return_data etc. We could also do that via return values but then we
            # would hold additional copied in memory. This is probably over the top but eh.
            self.augment_spatial(sample_idx,
                                 data[sample_idx],
                                 seg[sample_idx] if seg is not None else None,
                                 regr_target[sample_idx] if regr_target is not None else None)

        data_dict[self.data_key] = self.return_data
        if seg is not None:
            data_dict[self.seg_key] = self.return_seg
        if regr_target is not None:
            data_dict[self.regr_target_key] = self.return_regr_target

        del self.return_data, self.return_seg, self.return_regr_target
        gc.collect()
        return data_dict

    def augment_spatial(self, sample_idx: int, data_sample, seg_sample, regr_target_sample) -> None:
        do_deform = np.random.uniform() < self.p_el_per_sample
        do_scale = np.random.uniform() < self.p_scale_per_sample
        do_rot = np.random.uniform() < self.p_rot_per_sample

        if not any((do_deform, do_rot, do_scale)):
            margin = tuple(
                [pcdist - ps // 2 for pcdist, ps in zip(self.patch_center_dist_from_border, self.patch_size)])
            return_data, return_seg, return_regr_target = \
                crop(data_sample, seg_sample, regr_target_sample,
                     crop_size=self.patch_size, margins=margin, crop_type='random' if self.random_crop else 'center',
                     pad_modes=(self.border_mode_data, self.border_mode_seg, self.border_mode_regr_target),
                     pad_kwargs=({'constant_values': self.border_cval_data},
                                 {'constant_values': self.border_cval_seg},
                                 {'constant_values': self.border_cval_regr_target}))
            self.return_data[sample_idx] = return_data
            if return_seg is not None:
                self.return_seg[sample_idx] = return_seg
            if return_regr_target is not None:
                self.return_regr_target[sample_idx] = return_regr_target
        else:
            coordinates = self._create_coordinates(self.patch_size)
            if do_scale:
                coordinates = self.apply_random_scaling_to_coordinates(coordinates, self.p_independent_scale_per_axis,
                                                                       self.scale,
                                                                       False)  # coordinate mesh is already centered, no need to recenter it.

            if do_rot:
                coordinates = self.apply_random_rotation_to_coordinates(coordinates, self.p_rot_per_axis,
                                                                        self.angle_x, self.angle_y, self.angle_z)
            if do_deform:
                coordinates = self.apply_deformation_to_coordindates(coordinates, self.deformation_scale,
                                                                     self.deformation_magnitude, use_fft=False)

            # now place the patch in the correct location (it is currently zero-centered and now must be placed into the correct position in the inputs)
            patch_center = np.mean(coordinates, axis=tuple(range(1, len(coordinates.shape))))
            if self.random_crop:
                crop_center_in_image = np.array([np.random.uniform(pcd, ds - pcd) for ds, pcd in
                                                 zip(data_sample.shape[1:], self.patch_center_dist_from_border)])
            else:
                crop_center_in_image = np.array(data_sample.shape[1:]) / 2 - 0.5

            coordinates += - np.expand_dims(patch_center, axis=tuple(range(1, len(coordinates.shape)))) + \
                           np.expand_dims(crop_center_in_image, axis=tuple(range(1, len(coordinates.shape))))
            self.return_data[sample_idx] = self.interpolate_image(data_sample, coordinates)
            if seg_sample is not None:
                self.return_seg[sample_idx] = self.interpolate_seg(seg_sample, coordinates)
            if regr_target_sample is not None:
                self.return_regr_target[sample_idx] = self.interpolate_regtarget(regr_target_sample, coordinates)

    def interpolate_image(self, data: np.ndarray, coordinates: np.ndarray):
        result = np.zeros_like(data, shape=(data.shape[0], *coordinates.shape[1:]))
        for c in range(data.shape[0]):
            result[c] = map_coordinates(data[c], coordinates, order=self.order_data, mode=self.border_mode_data,
                                        cval=self.border_cval_data, prefilter=self.prefilter)
        return result

    def interpolate_seg(self, seg: np.ndarray, coordinates: np.ndarray):
        result = np.zeros_like(seg, shape=(seg.shape[0], *coordinates.shape[1:]))
        for c in range(seg.shape[0]):
            if self.order_seg == 0:
                result[c] = map_coordinates(seg[c], coordinates, order=self.order_seg,
                                            mode=self.border_mode_seg, cval=self.border_cval_seg,
                                            prefilter=self.prefilter)
            else:
                unique_values = np.sort(pd.unique(seg.ravel()))
                for v in unique_values:
                    res_new = map_coordinates((seg[c] == v).astype(np.float32), coordinates, order=self.order_seg,
                                              mode=self.border_mode_seg, cval=-1,
                                              prefilter=self.prefilter)
                    result[c][res_new >= 0.5] = v
                    result[c][res_new < 0] = self.border_cval_seg
        return result

    def interpolate_regtarget(self, regr_target: np.ndarray, coordinates: np.ndarray):
        result = np.zeros_like(regr_target, shape=(regr_target.shape[0], *coordinates.shape[1:]))
        for c in range(regr_target.shape[0]):
            result[c] = map_coordinates(regr_target[c], coordinates, order=self.order_regr_target,
                                        mode=self.border_mode_regr_target, cval=self.border_cval_regr_target,
                                        prefilter=self.prefilter)
        return result

    @staticmethod
    def apply_deformation_to_coordindates(coordinates: np.ndarray, deformation_scale: ScalarType,
                                          deformation_magnitude: ScalarType, use_fft: bool = False):
        image_shape = coordinates.shape[1:]
        for dim in range(coordinates.shape[0]):
            # sample a random deformation field
            random_deformation = np.random.uniform(-1, 1, size=image_shape)
            scale = sample_scalar(deformation_scale, image_shape, dim)
            assert scale > 0, 'deformation scale must be > 0'
            if use_fft:
                random_deformation = gaussian_filter(random_deformation, sigma=scale)
            else:
                random_deformation = np.fft.fftn(random_deformation)
                random_deformation = fourier_gaussian(random_deformation, scale)
                random_deformation = np.fft.ifftn(random_deformation).real
            magnitude = sample_scalar(deformation_magnitude, image_shape, dim, scale)
            random_deformation = random_deformation / np.max(np.abs(random_deformation)) * magnitude
            coordinates[dim] += random_deformation
        return coordinates

    @staticmethod
    def apply_random_rotation_to_coordinates(coordinates: np.ndarray, p_rot_per_axis: float, angle_x: ScalarType,
                                             angle_y: ScalarType, angle_z: ScalarType):
        image_shape = coordinates.shape[1:]
        a_x = sample_scalar(angle_x, image_shape, 0) if np.random.uniform() < p_rot_per_axis else 0
        # unbelievable that I am still using these old functions. Works, eh?
        if len(image_shape) == 3:
            a_y = sample_scalar(angle_y, image_shape, 1) if np.random.uniform() < p_rot_per_axis else 0
            a_z = sample_scalar(angle_z, image_shape, 2) if np.random.uniform() < p_rot_per_axis else 0
            return rotate_coords_3d(coordinates, a_x, a_y, a_z)
        elif len(image_shape) == 2:
            return rotate_coords_2d(coordinates, a_x)
        else:
            raise RuntimeError(f'Unsupported image dimension: {len(image_shape)}. Only 2D and 3D works, yo')

    @staticmethod
    def apply_random_scaling_to_coordinates(coordinates: np.ndarray, p_independent_scale_per_axis: float,
                                            scale_range: ScalarType, recenter_for_scaling: bool = False):
        if recenter_for_scaling:
            center = np.mean(coordinates, axis=([i for i in range(1, len(coordinates.shape))]))
            center = np.expand_dims(center, axis=[i for i in range(1, len(coordinates.shape))])
            coordinates -= center

        if np.random.uniform() < p_independent_scale_per_axis:
            for d in range(coordinates.shape[0]):
                scale = sample_scalar(scale_range, coordinates.shape[1:], d)
                assert scale > 0
                coordinates[d] *= scale
        else:
            # scale_range ScalaType must accept None as input for dim!
            scale = sample_scalar(scale_range, coordinates.shape[1:], None)
            assert scale > 0
            coordinates *= scale

        if recenter_for_scaling:
            coordinates += center

        return coordinates

    @staticmethod
    def _create_coordinates(shape: Tuple[int, ...]):
        tmp = tuple([np.arange(i) for i in shape])
        shape = np.array(shape)
        coords = np.array(np.meshgrid(*tmp, indexing='ij')).astype(np.float32)
        coords -= np.expand_dims((shape - 1) / 2., [i + 1 for i in range(len(shape))])
        return coords


if __name__ == '__main__':
    # a = np.zeros((3, 128, 256, 32))
    # a[:, 32:128-32, 48:256-48, 8:32-8] = 1
    # b = crop(a, crop_size=(100, 200, 24), crop_type='random', margins=999)
    image = '/media/isensee/raw_data/nnUNet_raw/Dataset017_AbdominalOrganSegmentation/imagesTr/img0005_0000.nii.gz'
    seg = '/media/isensee/raw_data/nnUNet_raw/Dataset017_AbdominalOrganSegmentation/labelsTr/img0005.nii.gz'
    import SimpleITK as sitk

    image = sitk.GetArrayFromImage(sitk.ReadImage(image))
    seg = sitk.GetArrayFromImage(sitk.ReadImage(seg))
    data_dict = {'data': image[None, None], 'seg': seg[None, None].astype(np.int8)}
    sptr = CleanerSpatialTransform(patch_size=(128, 128, 128), interpolation_order_seg=0, p_el_per_sample=1, p_rot_per_sample=0, border_cval_data=0,patch_center_dist_from_border=64, border_cval_seg=-1, border_mode_data='constant')
    ret = sptr(**deepcopy(data_dict))
    print(ret['seg'].min(), ret['seg'].max())
    from batchviewer import view_batch
    view_batch(ret['data'][0], ret['seg'][0])

