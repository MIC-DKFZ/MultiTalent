from __future__ import annotations
from multitalent.utilities.MultiTalent.MultiTalent_architecture import PlainConvUNet, ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from multitalent.utilities.network_initialization import InitWeights_He
from multitalent.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
import pydoc
import warnings
from typing import Union
from importlib.metadata import version

from multitalent.utilities.find_class_by_name import recursive_find_python_class
from batchgenerators.utilities.file_and_folder_operations import join



'''def get_MT_network_from_plans(plans_managers: dict,
                           dataset_jsons: dict,
                           configuration_manager: ConfigurationManager,
                           num_input_channels: int | list,
                           deep_supervision: bool = True):
    """
    we may have to change this in the future to accommodate other plans -> network mappings

    num_input_channels can differ depending on whether we do cascade. Its best to make this info available in the
    trainer rather than inferring it again from the plans here.
    """
    num_stages = len(configuration_manager.conv_kernel_sizes)

    dim = len(configuration_manager.conv_kernel_sizes[0])
    conv_op = convert_dim_to_conv_op(dim)

    segmentation_network_class_name = configuration_manager.UNet_class_name

    mapping = {
        'PlainConvUNet': PlainConvUNet,
        'ResidualEncoderUNet': ResidualEncoderUNet
    }
    kwargs = {
        'PlainConvUNet': {
            'conv_bias': True,
            'norm_op': get_matching_instancenorm(conv_op),
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None, 'dropout_op_kwargs': None,
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        },
        'ResidualEncoderUNet': {
            'conv_bias': True,
            'norm_op': get_matching_instancenorm(conv_op),
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None, 'dropout_op_kwargs': None,
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        }
    }
    assert segmentation_network_class_name in mapping.keys(), 'The network architecture specified by the plans file ' \
                                                              'is non-standard (maybe your own?). Yo\'ll have to dive ' \
                                                              'into either this ' \
                                                              'function (get_network_from_plans) or ' \
                                                              'the init of your nnUNetModule to accommodate that.'
    network_class = mapping[segmentation_network_class_name]

    conv_or_blocks_per_stage = {
        'n_conv_per_stage'
        if network_class != ResidualEncoderUNet else 'n_blocks_per_stage': configuration_manager.n_conv_per_stage_encoder,
        'n_conv_per_stage_decoder': configuration_manager.n_conv_per_stage_decoder
    }

    dataset_segheads = {}
    for key in plans_managers.keys():
        dataset_labelmanager = plans_managers[key].get_label_manager(dataset_jsons[key])
        dataset_segheads[key] = dataset_labelmanager.num_segmentation_heads
    # network class name!!
    model = network_class(
        input_channels=num_input_channels,
        n_stages=num_stages,
        features_per_stage=[min(configuration_manager.UNet_base_num_features * 2 ** i,
                                configuration_manager.unet_max_num_features) for i in range(num_stages)],
        conv_op=conv_op,
        kernel_sizes=configuration_manager.conv_kernel_sizes,
        strides=configuration_manager.pool_op_kernel_sizes,
        num_classes=dataset_segheads,
        deep_supervision=deep_supervision,
        **conv_or_blocks_per_stage,
        **kwargs[segmentation_network_class_name]
    )
    model.apply(InitWeights_He(1e-2))
    if network_class == ResidualEncoderUNet:
        model.apply(init_last_bn_before_add_to_0)
    return model'''


def get_MT_network_from_plans(arch_class_name, arch_kwargs, arch_kwargs_req_import, input_channels, output_channels,
                           allow_init=True, deep_supervision: Union[bool, None] = None):
    network_class = arch_class_name
    architecture_kwargs = dict(**arch_kwargs)
    for ri in arch_kwargs_req_import:
        if architecture_kwargs[ri] is not None:
            architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])

    if version('dynamic-network-architectures') == '0.3.1' or version('dynamic-network-architectures') == '0.2':
        if network_class == 'dynamic_network_architectures.architectures.residual_unet.ResidualEncoderUNet':
            # network_class = 'dynamic_network_architectures.architectures.unet.ResidualEncoderUNet'
            network_class = 'multitalent.utilities.MultiTalent.MultiTalent_architecture.ResidualEncoderUNet'
        elif network_class == 'dynamic_network_architectures.architectures.unet.ResidualEncoderUNet':
            network_class = 'multitalent.utilities.MultiTalent.MultiTalent_architecture.ResidualEncoderUNet'
        else:
            network_class = 'multitalent.utilities.MultiTalent.MultiTalent_architecture.PlainConvUNet'
    nw_class = pydoc.locate(network_class)
    # sometimes things move around, this makes it so that we can at least recover some of that
    if nw_class is None:
        warnings.warn(f'Network class {network_class} not found. Attempting to locate it within '
                      f'dynamic_network_architectures.architectures...')
        import dynamic_network_architectures
        nw_class = recursive_find_python_class(join(dynamic_network_architectures.__path__[0], "architectures"),
                                               network_class.split(".")[-1],
                                               'dynamic_network_architectures.architectures')
        if nw_class is not None:
            print(f'FOUND IT: {nw_class}')
        else:
            raise ImportError('Network class could not be found, please check/correct your plans file')



    if deep_supervision is not None and 'deep_supervision' not in arch_kwargs.keys():
        arch_kwargs['deep_supervision'] = deep_supervision

    network = nw_class(
        input_channels=input_channels,
        num_classes=output_channels,
        **architecture_kwargs)

    if hasattr(network, 'initialize') and allow_init:
        network.apply(network.initialize)

    return network
