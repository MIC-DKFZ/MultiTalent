import random

import torch
from typing import Union, Tuple
from threadpoolctl import threadpool_limits
from batchgenerators.dataloading.data_loader import DataLoader
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from multitalent.training.dataloading.nnunet_dataset import nnUNetDataset
from multitalent.utilities.label_handling.label_handling import LabelManager

import time
from multitalent.training.dataloading.base_data_loader import nnUNetDataLoaderBase
from typing import Union, Tuple
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from multitalent.training.dataloading.nnunet_dataset import nnUNetDataset
from multitalent.utilities.label_handling.label_handling import LabelManager

class nnUNetDataLoader3D_MT(nnUNetDataLoaderBase):
    def __init__(self,
                 data: nnUNetDataset,
                 batch_size: int,
                 patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 final_patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 label_manager: LabelManager,
                 oversample_foreground_percent: float = 0.0,
                 sampling_probabilities: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 pad_sides: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 probabilistic_oversampling: bool = False,
                 transforms=None,
                 labelmapping: list = None):
        super().__init__(data, batch_size, patch_size, final_patch_size, label_manager, oversample_foreground_percent, sampling_probabilities, pad_sides,probabilistic_oversampling, transforms)
        self.labelmapping = labelmapping
    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        case_properties = []

        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            force_fg = self.get_do_oversample(j)

            data, seg, properties = self._data.load_case(i)
            case_properties.append(properties)

            # If we are doing the cascade then the segmentation from the previous stage will already have been loaded by
            # self._data.load_case(i) (see nnUNetDataset.load_case)
            shape = data.shape[1:]
            dim = len(shape)

            #for datasets with more than x classes we want to split up the classes --> need to adapt classes for oversampling
            if self.labelmapping is not None:
                prop_loc = {}
                for i in self.labelmapping:
                    prop_loc[i[1]] = properties['class_locations'][i[1]]
            else:
                prop_loc = properties['class_locations']
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, prop_loc)

            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_lbs = np.clip(bbox_lbs, a_min=0, a_max=None)
            valid_bbox_ubs = np.minimum(shape, bbox_ubs)

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)
            this_slice = tuple([slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            data = data[this_slice]

            this_slice = tuple([slice(0, seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            seg = seg[this_slice]

            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
            padding = ((0, 0), *padding)
            data_all[j] = np.pad(data, padding, 'constant', constant_values=0)
            seg_all[j] = np.pad(seg, padding, 'constant', constant_values=-1)

        if self.labelmapping is not None:
            #batchsize is always 1 fo MT
            new_seg = np.zeros(seg_all.shape, dtype=seg_all.dtype)
            for map in self.labelmapping:
                new_seg[seg_all==map[0]] = map[1]
            seg_all = new_seg

        if self.transforms is not None:
            if torch is not None:
                torch_nthreads = torch.get_num_threads()
                torch.set_num_threads(1)
            with threadpool_limits(limits=1, user_api=None):
                data_all = torch.from_numpy(data_all).float()
                seg_all = torch.from_numpy(seg_all).to(torch.int16)
                images = []
                segs = []
                for b in range(self.batch_size):
                    tmp = self.transforms(**{'image': data_all[b], 'segmentation': seg_all[b]})
                    images.append(tmp['image'])
                    segs.append(tmp['segmentation'])
                data_all = torch.stack(images)
                seg_all = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
                del segs, images
            if torch is not None:
                torch.set_num_threads(torch_nthreads)

            return {'data': data_all, 'seg': seg_all, 'properties': case_properties, 'keys': selected_keys}



        return {'data': data_all, 'seg': seg_all, 'keys': selected_keys}

class nnUNetDataLoader3D_MTall(DataLoader):
    def __init__(self,
                 data: nnUNetDataset,
                 batch_size: int,
                 patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 final_patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 label_managers: dict(),
                 oversample_foreground_percent: float = 0.0,
                 sampling_probabilities: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 pad_sides: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 probabilistic_oversampling: bool = False,
                 transforms=None,
                 labelmapping: dict = None,
                 input_channels: dict = None):
        super().__init__(data, batch_size, 1, None, True, False, True, sampling_probabilities)
        self.labelmapping = labelmapping
        self.indices = list(data.keys())

        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        # need_to_pad denotes by how much we need to pad the data so that if we sample a patch of size final_patch_size
        # (which is what the network will get) these patches will also cover the border of the images
        self.need_to_pad = (np.array(patch_size) - np.array(final_patch_size)).astype(int)
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.num_channels = None
        self.pad_sides = pad_sides
        #self.datashape can be different due to multi channel input
        self.input_channels = input_channels
        self.data_shape, self.seg_shape = self.determine_shapes()
        self.sampling_probabilities = sampling_probabilities
        self.annotated_classes_key = {}
        self.has_ignore = {}
        for id in label_managers.keys():
            self.annotated_classes_key[id] = tuple(label_managers[id].all_labels)
            self.has_ignore[id] = label_managers[id].has_ignore_label
        self.get_do_oversample = self._oversample_last_XX_percent if not probabilistic_oversampling \
            else self._probabilistic_oversampling
        self.transforms = transforms
    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        #data all size depends on selected case id
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        case_properties = []

        if len(selected_keys) > 1:
            raise RuntimeError("MultiTalent dataloader only works with batch size 2")
        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            force_fg = self.get_do_oversample(j)

            d_id = i[7:10]
            possible_ids = []
            for id in self.annotated_classes_key.keys():
                if id.startswith(d_id):
                    possible_ids.append(id)

            case_id = random.choice(possible_ids)
            shape = [1, self.input_channels[case_id], self.data_shape[-3], self.data_shape[-2], self.data_shape[-1]]
            data_all = np.zeros(shape, dtype=np.float32)

            data, seg, properties = self._data.load_case(i)
            case_properties.append(properties)

            # If we are doing the cascade then the segmentation from the previous stage will already have been loaded by
            # self._data.load_case(i) (see nnUNetDataset.load_case)
            shape = data.shape[1:]
            dim = len(shape)

            #for datasets with more than x classes we want to split up the classes --> need to adapt classes for oversampling
            if self.labelmapping[case_id] is not None:
                prop_loc = {}
                for i in self.labelmapping[case_id]:
                    prop_loc[i[1]] = properties['class_locations'][i[1]]
            else:
                prop_loc = properties['class_locations']
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, case_id, prop_loc)

            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_lbs = np.clip(bbox_lbs, a_min=0, a_max=None)
            valid_bbox_ubs = np.minimum(shape, bbox_ubs)

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)
            this_slice = tuple([slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            data = data[this_slice]

            this_slice = tuple([slice(0, seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            seg = seg[this_slice]

            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
            padding = ((0, 0), *padding)

            data_all[j] = np.pad(data, padding, 'constant', constant_values=0)
            seg_all[j] = np.pad(seg, padding, 'constant', constant_values=-1)
            # start = time.time()
            if self.labelmapping[case_id] is not None:
                #batchsize is always 1 fo MT
                new_seg = np.zeros(seg_all.shape, dtype=seg_all.dtype)
                for map in self.labelmapping[case_id]:
                    new_seg[seg_all==map[0]] = map[1]
                seg_all = new_seg
                # print(time.time()-start)


        if self.transforms[case_id] is not None:
            if torch is not None:
                torch_nthreads = torch.get_num_threads()
                torch.set_num_threads(1)
            with threadpool_limits(limits=1, user_api=None):
                data_all = torch.from_numpy(data_all).float()
                seg_all = torch.from_numpy(seg_all).to(torch.int16)
                images = []
                segs = []
                for b in range(self.batch_size):
                    tmp = self.transforms[case_id](**{'image': data_all[b], 'segmentation': seg_all[b]})
                    images.append(tmp['image'])
                    segs.append(tmp['segmentation'])
                data_all = torch.stack(images)
                seg_all = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
                del segs, images
            if torch is not None:
                torch.set_num_threads(torch_nthreads)
            return {'data': data_all, 'seg': seg_all, 'keys': selected_keys, 'id': case_id}


        return {'data': data_all, 'seg': seg_all, 'keys': selected_keys, 'id': case_id}

    def _oversample_last_XX_percent(self, sample_idx: int) -> bool:
        """
        determines whether sample sample_idx in a minibatch needs to be guaranteed foreground
        """
        return not sample_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))

    def _probabilistic_oversampling(self, sample_idx: int) -> bool:
        # print('YEAH BOIIIIII')
        return np.random.uniform() < self.oversample_foreground_percent

    def determine_shapes(self):
        # load one case
        data, seg, properties = self._data.load_case(self.indices[0])
        num_color_channels = data.shape[0]

        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, seg.shape[0], *self.patch_size)
        return data_shape, seg_shape

    def get_bbox(self, data_shape: np.ndarray, force_fg: bool, id: str, class_locations: Union[dict, None],
                 overwrite_class: Union[int, Tuple[int, ...]] = None, verbose: bool = False):
        # in dataloader 2d we need to select the slice prior to this and also modify the class_locations to only have
        # locations for the given slice
        need_to_pad = self.need_to_pad.copy()
        dim = len(data_shape)

        for d in range(dim):
            # if case_all_data.shape + need_to_pad is still < patch size we need to pad more! We pad on both sides
            # always
            if need_to_pad[d] + data_shape[d] < self.patch_size[d]:
                need_to_pad[d] = self.patch_size[d] - data_shape[d]

        # we can now choose the bbox from -need_to_pad // 2 to shape - patch_size + need_to_pad // 2. Here we
        # define what the upper and lower bound can be to then sample form them with np.random.randint
        lbs = [- need_to_pad[i] // 2 for i in range(dim)]
        ubs = [data_shape[i] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - self.patch_size[i] for i in range(dim)]

        # if not force_fg then we can just sample the bbox randomly from lb and ub. Else we need to make sure we get
        # at least one of the foreground classes in the patch
        if not force_fg and not self.has_ignore[id]:
            bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]
            # print('I want a random location')
        else:
            if not force_fg and self.has_ignore[id]:
                selected_class = self.annotated_classes_key[id]
                if len(class_locations[selected_class]) == 0:
                    # no annotated pixels in this case. Not good. But we can hardly skip it here
                    print('Warning! No annotated pixels in image!')
                    selected_class = None
                # print(f'I have ignore labels and want to pick a labeled area. annotated_classes_key: {self.annotated_classes_key}')
            elif force_fg:
                assert class_locations is not None, 'if force_fg is set class_locations cannot be None'
                if overwrite_class is not None:
                    assert overwrite_class in class_locations.keys(), 'desired class ("overwrite_class") does not ' \
                                                                      'have class_locations (missing key)'
                # this saves us a np.unique. Preprocessing already did that for all cases. Neat.
                # class_locations keys can also be tuple
                eligible_classes_or_regions = [i for i in class_locations.keys() if len(class_locations[i]) > 0]

                # if we have annotated_classes_key locations and other classes are present, remove the annotated_classes_key from the list
                # strange formulation needed to circumvent
                # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
                tmp = [i == self.annotated_classes_key[id] if isinstance(i, tuple) else False for i in eligible_classes_or_regions]
                if any(tmp):
                    if len(eligible_classes_or_regions) > 1:
                        eligible_classes_or_regions.pop(np.where(tmp)[0][0])

                if len(eligible_classes_or_regions) == 0:
                    # this only happens if some image does not contain foreground voxels at all
                    selected_class = None
                    if verbose:
                        print('case does not contain any foreground classes')
                else:
                    # I hate myself. Future me aint gonna be happy to read this
                    # 2022_11_25: had to read it today. Wasn't too bad
                    selected_class = eligible_classes_or_regions[np.random.choice(len(eligible_classes_or_regions))] if \
                        (overwrite_class is None or (overwrite_class not in eligible_classes_or_regions)) else overwrite_class
                # print(f'I want to have foreground, selected class: {selected_class}')
            else:
                raise RuntimeError('lol what!?')
            voxels_of_that_class = class_locations[selected_class] if selected_class is not None else None

            if voxels_of_that_class is not None and len(voxels_of_that_class) > 0:
                selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]
                # selected voxel is center voxel. Subtract half the patch size to get lower bbox voxel.
                # Make sure it is within the bounds of lb and ub
                # i + 1 because we have first dimension 0!
                bbox_lbs = [max(lbs[i], selected_voxel[i + 1] - self.patch_size[i] // 2) for i in range(dim)]
            else:
                # If the image does not contain any foreground classes, we fall back to random cropping
                bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]

        bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(dim)]

        return bbox_lbs, bbox_ubs

if __name__ == '__main__':
    folder = '/media/fabian/data/nnUNet_preprocessed/Dataset002_Heart/3d_fullres'
    ds = nnUNetDataset(folder, 0)  # this should not load the properties!
    dl = nnUNetDataLoader3D_MT(ds, 5, (16, 16, 16), (16, 16, 16), 0.33, None, None)
    a = next(dl)
