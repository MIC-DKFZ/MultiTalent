#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from copy import deepcopy

import torch
from torch import nn
from torch.nn import Identity
import numpy as np


def _maybe_convert_scalar_to_list(conv_op, scalar):
    if not isinstance(scalar, (tuple, list, np.ndarray)):
        if conv_op == nn.Conv2d:
            return [scalar] * 2
        elif conv_op == nn.Conv3d:
            return [scalar] * 3
        elif conv_op == nn.Conv1d:
            return [scalar] * 1
        else:
            raise RuntimeError("Invalid conv op: %s" % str(conv_op))
    else:
        return scalar


def _get_matching_avgPool(conv_op):
    if conv_op == nn.Conv2d:
        return nn.AvgPool2d
    elif conv_op == nn.Conv3d:
        return nn.AvgPool3d
    elif conv_op == nn.Conv1d:
        return nn.AvgPool1d
    else:
        raise RuntimeError("Invalid conv op: %s" % str(conv_op))


class ConvDropoutNormReLU(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, network_props):
        """
        if network_props['dropout_op'] is None then no dropout
        if network_props['norm_op'] is None then no norm
        :param input_channels:
        :param output_channels:
        :param kernel_size:
        :param network_props:
        """
        super(ConvDropoutNormReLU, self).__init__()

        kernel_size = _maybe_convert_scalar_to_list(network_props['conv_op'], kernel_size)

        network_props = deepcopy(network_props)  # network_props is a dict and mutable, so we deepcopy to be safe.

        self.conv = network_props['conv_op'](input_channels, output_channels, kernel_size,
                                             padding=[(i - 1) // 2 for i in kernel_size],
                                             **network_props['conv_op_kwargs'])

        # maybe dropout
        if network_props['dropout_op'] is not None:
            self.do = network_props['dropout_op'](**network_props['dropout_op_kwargs'])
        else:
            self.do = Identity()

        if network_props['norm_op'] is not None:
            self.norm = network_props['norm_op'](output_channels, **network_props['norm_op_kwargs'])
        else:
            self.norm = Identity()

        self.nonlin = network_props['nonlin'](**network_props['nonlin_kwargs'])

        self.all = nn.Sequential(self.conv, self.do, self.norm, self.nonlin)

    def forward(self, x):
        return self.all(x)


class StackedConvLayers(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, network_props, num_convs, first_stride=None):
        """
        if network_props['dropout_op'] is None then no dropout
        if network_props['norm_op'] is None then no norm
        :param input_channels:
        :param output_channels:
        :param kernel_size:
        :param network_props:
        """
        super(StackedConvLayers, self).__init__()

        network_props = deepcopy(network_props)  # network_props is a dict and mutable, so we deepcopy to be safe.
        network_props_first = deepcopy(network_props)

        if first_stride is not None:
            network_props_first['conv_op_kwargs']['stride'] = first_stride

        self.convs = nn.Sequential(
            ConvDropoutNormReLU(input_channels, output_channels, kernel_size, network_props_first),
            *[ConvDropoutNormReLU(output_channels, output_channels, kernel_size, network_props) for _ in
              range(num_convs - 1)]
        )

    def forward(self, x):
        return self.convs(x)


class BasicResidualBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, props, stride=None, use_avgpool_in_skip=False):
        """
        This is the conv bn nonlin conv bn nonlin kind of block
        :param in_planes:
        :param out_planes:
        :param props:
        :param override_stride:
        :param use_avgpool_in_skip: if True, will use nn.AvgPoolNd -> nn.ConvNd (1x1(x1)) in the skip connection to
        reduce the feature map size. If False, it will simply use strided nn.ConvNd (1x1(x1)) which throws away
        information
        """
        super().__init__()
        # props is a dict and we don't want anything happening to it
        props = deepcopy(props)

        # we ignore and stride from props['conv_op_kwargs']
        del props['conv_op_kwargs']['stride']

        kernel_size = _maybe_convert_scalar_to_list(props['conv_op'], kernel_size)

        # check if stride is OK, make sure stride is a list of
        if stride is not None:
            if isinstance(stride, (tuple, list)):
                # replacing all None entries with 1
                stride = [i if i is not None else 1 for i in stride]
            else:
                stride = _maybe_convert_scalar_to_list(props['conv_op'], stride)
        else:
            stride = _maybe_convert_scalar_to_list(props['conv_op'], 1)

        self.stride = stride

        self.kernel_size = kernel_size
        self.props = props
        self.out_planes = out_planes
        self.in_planes = in_planes

        self.conv1 = props['conv_op'](in_planes, out_planes,
                                      kernel_size=kernel_size,
                                      padding=[(i - 1) // 2 for i in kernel_size],
                                      stride=stride,
                                      **props['conv_op_kwargs'])
        self.norm1 = props['norm_op'](out_planes, **props['norm_op_kwargs'])
        self.nonlin1 = props['nonlin'](**props['nonlin_kwargs'])

        if props['dropout_op_kwargs']['p'] != 0:
            self.dropout = props['dropout_op'](**props['dropout_op_kwargs'])
        else:
            self.dropout = Identity()

        self.conv2 = props['conv_op'](out_planes, out_planes,
                                      kernel_size=kernel_size,
                                      padding=[(i - 1) // 2 for i in kernel_size],
                                      stride=1,
                                      **props['conv_op_kwargs'])
        self.norm2 = props['norm_op'](out_planes, **props['norm_op_kwargs'])
        self.nonlin2 = props['nonlin'](**props['nonlin_kwargs'])

        if (stride is not None and any([i != 1 for i in stride])) or (in_planes != out_planes):
            if use_avgpool_in_skip:
                ops = []
                # we only need to pool if any of the strides is != 1
                if any([i != 1 for i in stride]):
                    ops.append(_get_matching_avgPool(props['conv_op'])(stride, stride))
                # 1x1(x1) conv.
                # Todo: Do we really need this when all we do is change the resolution and not the number of feature maps?
                ops.append(props['conv_op'](in_planes, out_planes, kernel_size=1,
                                            padding=0,
                                            stride=1,
                                            bias=False))
                ops.append(props['norm_op'](out_planes, **props['norm_op_kwargs']))

                self.downsample_skip = nn.Sequential(*ops)
            else:
                stride_here = stride if stride is not None else 1
                self.downsample_skip = nn.Sequential(props['conv_op'](in_planes, out_planes,
                                                                      kernel_size=1,
                                                                      padding=0,
                                                                      stride=stride_here,
                                                                      bias=False),
                                                     props['norm_op'](out_planes, **props['norm_op_kwargs']))
        else:
            self.downsample_skip = lambda x: x

    def forward(self, x):
        residual = x

        out = self.dropout(self.conv1(x))
        out = self.nonlin1(self.norm1(out))

        out = self.norm2(self.conv2(out))

        residual = self.downsample_skip(residual)

        out += residual

        return self.nonlin2(out)


class ResidualBottleneckBlock(nn.Module):
    def __init__(self, in_planes, bottleneck_planes, out_planes, kernel_size, props, stride=None):
        """
        conv, norm, nonlin, conv, norm, nonlin, conv, norm (add) nonlin

        feature map numbers (output!) (input_to_this_block->conv1 -> conv2 -> conv3):
        in_planes->bottleneck_planes -> bottleneck_planes -> out_planes

        stride in the feature extraction branch happens in the middle convolution (typically 3x3) and not the 1x1 one.
        stride in the skip connection is implemented as avgpool -> 1x1 conv
        (also see
        He, T., Zhang, Z., Zhang, H., Zhang, Z., Xie, J., and Li, M.
        Bag of tricks for image classification with convolutional neural networks. arXiv preprint arXiv:1812.01187,
        2018.)

        :param in_planes:
        :param out_planes:
        :param bottleneck_planes:
        :param kernel_size:
        :param props:
        :param override_stride:
        """
        super().__init__()

        # props is a dict and we don't want anything happening to it
        props = deepcopy(props)

        if props['dropout_op_kwargs'] is not None and props['dropout_op_kwargs']['p'] > 0:
            raise NotImplementedError("ResidualBottleneckBlock does not yet support dropout!")

        kernel_size = _maybe_convert_scalar_to_list(props['conv_op'], kernel_size)
        self.kernel_size = kernel_size

        # we ignore and stride from props['conv_op_kwargs']
        del props['conv_op_kwargs']['stride']

        # check if stride is OK, make sure stride is a list of
        if stride is not None:
            if isinstance(stride, (tuple, list)):
                # replacing all None entries with 1
                stride = [i if i is not None else 1 for i in stride]
            else:
                stride = _maybe_convert_scalar_to_list(props['conv_op'], stride)
        else:
            stride = _maybe_convert_scalar_to_list(props['conv_op'], 1)

        self.stride = stride

        self.props = props
        self.out_planes = out_planes
        self.in_planes = in_planes
        self.bottleneck_planes = bottleneck_planes

        # first conv is a 1x1(x1) conv that goes from in_planes -> bottleneck_planes
        self.conv1 = props['conv_op'](in_planes, self.bottleneck_planes,
                                      kernel_size=1,
                                      padding=0,
                                      stride=1,
                                      **props['conv_op_kwargs'])
        self.norm1 = props['norm_op'](self.bottleneck_planes, **props['norm_op_kwargs'])
        self.nonlin1 = props['nonlin'](**props['nonlin_kwargs'])

        # second conv is a kernel_size conv (probably 3x3(x3)) that goes from bottleneck_planes -> bottleneck_planes
        # if stride is not None then it will be applied here. Note that stride > kernel_size will result in missing
        # information (in that case make sure to set kernel_size to an appropriate size)
        self.conv2 = props['conv_op'](self.bottleneck_planes, self.bottleneck_planes,
                                      kernel_size=kernel_size,
                                      padding=[(i - 1) // 2 for i in kernel_size],
                                      stride=stride,
                                      **props['conv_op_kwargs'])
        self.norm2 = props['norm_op'](self.bottleneck_planes, **props['norm_op_kwargs'])
        self.nonlin2 = props['nonlin'](**props['nonlin_kwargs'])

        # third conv is a 1x1(x1) conv that goes from bottleneck_planes -> out_planes
        self.conv3 = props['conv_op'](self.bottleneck_planes, out_planes,
                                      kernel_size=1,
                                      padding=0,
                                      stride=1,
                                      **props['conv_op_kwargs'])
        self.norm3 = props['norm_op'](out_planes, **props['norm_op_kwargs'])
        self.nonlin3 = props['nonlin'](**props['nonlin_kwargs'])

        # this triggers the 1x1(x1) conv (+ downsampling) in the skip path
        if (stride is not None and any([i != 1 for i in stride])) or (in_planes != out_planes):
            ops = []
            # we only need to pool if any of the strides is != 1
            if any([i != 1 for i in stride]):
                ops.append(_get_matching_avgPool(props['conv_op'])(stride, stride))
            # 1x1(x1) conv.
            # Todo: Do we really need this when all we do is change the resolution and not the number of feature maps?
            ops.append(props['conv_op'](in_planes, out_planes, kernel_size=1,
                                        padding=0,
                                        stride=1,
                                        bias=False))
            ops.append(props['norm_op'](out_planes, **props['norm_op_kwargs']))

            self.downsample_skip = nn.Sequential(*ops)
        else:
            self.downsample_skip = lambda x: x

    def forward(self, x):
        residual = x

        out = self.nonlin1(self.norm1(self.conv1(x)))
        out = self.nonlin2(self.norm2(self.conv2(out)))
        out = self.norm3(self.conv3(out))

        residual = self.downsample_skip(residual)

        out += residual

        return self.nonlin3(out)


class ResidualLayer(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, network_props, num_blocks, first_stride=None,
                 block=BasicResidualBlock, block_kwargs=None):
        super().__init__()

        if block_kwargs is None:
            block_kwargs = {}
        network_props = deepcopy(network_props)  # network_props is a dict and mutable, so we deepcopy to be safe.

        if block == ResidualBottleneckBlock:
            self.convs = []
            self.convs.append(block(input_channels, output_channels, output_channels * 4, kernel_size, network_props,
                                    first_stride, **block_kwargs))
            self.convs += [block(output_channels * 4, output_channels, output_channels * 4, kernel_size, network_props,
                                 **block_kwargs) for _ in range(num_blocks - 1)]
            self.convs = nn.Sequential(*self.convs)
            self.output_channels = output_channels * 4
        else:
            self.convs = nn.Sequential(
                block(input_channels, output_channels, kernel_size, network_props, first_stride, **block_kwargs),
                *[block(output_channels, output_channels, kernel_size, network_props, **block_kwargs) for _ in
                  range(num_blocks - 1)]
            )

            self.output_channels = output_channels

    def forward(self, x):
        return self.convs(x)

