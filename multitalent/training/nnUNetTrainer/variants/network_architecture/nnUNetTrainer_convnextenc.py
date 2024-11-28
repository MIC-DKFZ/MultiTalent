try:
    from dynamic_network_architectures.architectures.unet import ConvNextEncoderUNet
    from dynamic_network_architectures.building_blocks.convnext_encoder import ConvNextEncoder_standardconvblockstart
    from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_instancenorm
except ImportError:
    ConvNextEncoderUNet = None
from torch import nn

from multitalent.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from multitalent.utilities.network_initialization import InitWeights_He
from multitalent.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager


class nnUNetTrainer_convnextenc_regularconvblock(nnUNetTrainer):

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        num_stages = len(configuration_manager.conv_kernel_sizes)

        dim = len(configuration_manager.conv_kernel_sizes[0])
        conv_op = convert_dim_to_conv_op(dim)

        label_manager = plans_manager.get_label_manager(dataset_json)

        model = ConvNextEncoderUNet(num_input_channels, num_stages,
                                    [min(configuration_manager.UNet_base_num_features * 2 ** i,
                                         configuration_manager.unet_max_num_features) for i in range(num_stages)],
                                    conv_op, configuration_manager.conv_kernel_sizes,
                                    configuration_manager.pool_op_kernel_sizes,
                                    configuration_manager.n_conv_per_stage_encoder,
                                                       label_manager.num_segmentation_heads,
                                    configuration_manager.n_conv_per_stage_decoder, True,
                                    get_matching_instancenorm(conv_op),
                                    {'eps': 1e-5, 'affine': True}, nn.Identity, {}, nn.LeakyReLU, {'inplace': True},
                                    drop_path_rate_encoder=0, deep_supervision=enable_deep_supervision,
                                    encoder=ConvNextEncoder_standardconvblockstart)
        model.decoder.apply(InitWeights_He(1e-2))
        return model


