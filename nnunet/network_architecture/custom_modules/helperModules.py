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
import torch
from torch import nn
from torch.nn import functional


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, input):
        return input


class MyGroupNorm(nn.GroupNorm):
    def __init__(self, num_channels, eps=1e-5, affine=True, num_groups=8):
        super(MyGroupNorm, self).__init__(num_groups, num_channels, eps, affine)


class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=True):
        super(Upsample, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        if x.dtype == torch.half:
            return functional.interpolate(x.float(), size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                                          align_corners=self.align_corners).half()
        else:
            return functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                                          align_corners=self.align_corners)
