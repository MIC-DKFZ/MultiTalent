# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany  # noqa: E501
# SPDX-License-Identifier: Apache-2.0


import copy
from typing import Optional, Sequence, Type, Union, List
try:
    from nndet.nn.backbone.abstract import AbstractBackbone
    from nndet.nn.layers.wrapper import Generator
    from nndet.nn.neck.abstract import AbstractNeck
    from nndet.utils.typing import CONVSEQ
    from nndet.nn.backbone.blueprints.conv import ConvBackbone
    from nndet.nn.neck.fpn import UFPN
    from functools import partial
    from nndet.nn.layers.conv.instance import ConvInstanceLReLU
    from nndet.nn.layers.conv.group import ConvGroupLReLU
    from nndet.nn.layers.initializer import InitHeV2
except ImportError:
    AbstractBackbone = None
from torch import nn
import torch
from dynamic_network_architectures.initialization.weight_init import InitWeights_He



class DetSegModel_multiheads(nn.Module):
    def __init__(self, arch_init_kwargs, num_input_channels, num_output_channels, enable_deep_supervision):

        super(DetSegModel_multiheads, self).__init__()
        if AbstractBackbone is None:
            raise ImportError("nndet is required to run this trainer")
        self.backbone_cls: Type[AbstractBackbone] = ConvBackbone  #: define class for backbone
        self.backbone_conv_cls: Type[CONVSEQ] = partial(ConvInstanceLReLU, initializer=InitHeV2(
            mode="fan_out"))  #: conv class used for backbone

        self.neck_cls: Type[AbstractNeck] = UFPN  #: define class for neck
        self.neck_conv_cls: Type[CONVSEQ] = partial(ConvGroupLReLU, initializer=InitHeV2(mode="fan_out"))

        self.conv_kernel_sizes = arch_init_kwargs['kernel_sizes']
        self.pool_op_kernel_sizes = arch_init_kwargs['strides']
        self.base_num_features = arch_init_kwargs['features_per_stage'][0]
        self.num_levels = len(self.conv_kernel_sizes) - 1
        self.decoder_levels = [i for i in range(2, self.num_levels)]
        self.fpn_output_channels = 128
        self.level_cfgs = []
        self.input_channels = num_input_channels

        self.num_conv = arch_init_kwargs["n_conv_per_stage"]
        self.deep_supervision = enable_deep_supervision
        self.num_output_channels = num_output_channels

        self.backbone: torch.nn.Module = self.create_backbone()
        self.decoder: torch.nn.Module = self.create_neck()
        self.num_deepsupervision_stages = self.decoder.compute_output_channels()
        self.seg_layer: Union[torch.nn.Module, torch.nn.ModuleList] = self.create_segmentation_layer(
            in_channels=self.num_deepsupervision_stages, out_channels=num_output_channels)

    def create_level_cfgs(self):
        ###set level configs
        level_cfgs = []
        for i in range(
                self.num_levels):  # for i in range(self.num_levels+1) -> otherwise no _cfg is created for last stage
            _cfg = {
                "kernel": self.conv_kernel_sizes[i],
                "num_conv": self.num_conv[i],
                "kwargs": {},
            }
            if i > 0:
                _cfg["stride"] = self.pool_op_kernel_sizes[i]
            level_cfgs.append(_cfg)
        return level_cfgs

    def create_backbone(self):
        conv_bb = Generator(self.backbone_conv_cls, len(self.conv_kernel_sizes[0]))
        backbone = self.backbone_cls(
            conv=conv_bb,
            in_channels=self.input_channels,
            start_channels=self.base_num_features,
            pooling_mode="conv_kernel",
            max_channels=320,
            stem_cfg={},
            level_cfgs=self.create_level_cfgs())
        return backbone

    def create_neck(self):
        conv_neck = Generator(self.neck_conv_cls, len(self.conv_kernel_sizes[0]))
        decoder = self.neck_cls(
            conv=conv_neck,
            conv_kernels=self.conv_kernel_sizes,
            relative_strides=self.backbone.get_relative_strides(),
            upsampling_mode="transpose",
            in_channels=self.backbone.get_channels(),  # in_channels=self.backbone.get_channels(),
            first_decoder_level=min(self.decoder_levels),
            last_decoder_level=max(self.decoder_levels),
            fpn_out_channels=128)
        return decoder

    def create_segmentation_layer(self, in_channels: List[int], out_channels: dict):

        '''if self.deep_supervision:
            seg_layers = [nn.Conv3d(i, self.label_manager.num_segmentation_heads, 1) for i in in_channels]
            seg_layer = torch.nn.ModuleList(seg_layers)
        else:
            seg_layer = nn.Conv3d(in_channels[0], self.label_manager.num_segmentation_heads, 1)'''
        seg_layers= {}
        for id in out_channels.keys():
            seg_layers[id] = []
            for i in in_channels:
                seg_layers[id].append(nn.Conv3d(i,out_channels[id], 1))
            seg_layers[id] = torch.nn.ModuleList(seg_layers[id])
        seg_layers = nn.ModuleDict(seg_layers)

        return seg_layers

    def forward(self, x: list[torch.Tensor], ids:list) -> Union[torch.Tensor, List[torch.Tensor]]:
        x_in = None
        for b in x:
            if x_in is None:
                x_in = b
            else:
                x_in = torch.cat((x_in, b), dim=0)
        encoded = self.backbone(x_in)
        # print([encoded_output.shape for encoded_output in encoded])

        decoded = self.decoder(encoded)

        segmentation_outputs = []
        for c,id in enumerate(ids):
            segmentation_outputs.append([])
            for level_idx in range(len(self.num_deepsupervision_stages)):
                # from IPython import embed; embed();
                segmentation_outputs[c].append(self.seg_layer[id][level_idx](torch.unsqueeze(decoded[level_idx][c], dim=0)))

        if not self.deep_supervision:
            seg_output = []
            for i in range(len(segmentation_outputs)):
                seg_output[i] = segmentation_outputs[i][0]
        else:
            seg_output = segmentation_outputs
        # if not self.deep_supervision:
        # segmentation_outputs = self.seg_layer(decoded[0])
        # else:
        # segmentation_outputs = []
        # for level_idx, seg_head in enumerate(self.seg_layer):
        # segmentation_outputs.append(seg_head(decoded[level_idx]))
        # print([segmentation_output.shape for segmentation_output in segmentation_outputs])

        return seg_output

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)
