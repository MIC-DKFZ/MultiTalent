from typing import Union, Tuple, List
from multitalent.training.nnUNetTrainer.project_specific.MultiTalent_native.MultiTalent_trainer import MultiTalent_trainer_4000ep
from multitalent.utilities.MultiTalent.MultiTalent_meets_nndet.build_nndet_model import DetSegModel_multiheads
from torch import autocast, nn

class MultiTalent_meets_nndet_trainer_4000ep(MultiTalent_trainer_4000ep):
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int|dict,
                                   num_output_channels: dict,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        """
        This is where you build the architecture according to the plans. There is no obligation to use
        get_network_from_plans, this is just a utility we use for the nnU-Net default architectures. You can do what
        you want. Even ignore the plans and just return something static (as long as it can process the requested
        patch size)
        but don't bug us with your bugs arising from fiddling with this :-P
        This is the function that is called in inference as well! This is needed so that all network architecture
        variants can be loaded at inference time (inference will use the same nnUNetTrainer that was used for
        training, so if you change the network architecture during training by deriving a new trainer class then
        inference will know about it).

        If you need to know how many segmentation outputs your custom architecture needs to have, use the following snippet:
        > label_manager = plans_manager.get_label_manager(dataset_json)
        > label_manager.num_segmentation_heads
        (why so complicated? -> We can have either classical training (classes) or regions. If we have regions,
        the number of outputs is != the number of classes. Also there is the ignore label for which no output
        should be generated. label_manager takes care of all that for you.)

        """


        # need to be adapted for MT
        network = DetSegModel_multiheads(
            arch_init_kwargs,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision=enable_deep_supervision)

        return network


