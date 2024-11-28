import torch
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


def load_pretrained_weights_MT(network, fname, verbose=False, stemid:str =None, dublicate:str = None):
    """
    Transfers all weights between matching keys in state_dicts. matching is done by name and we only transfer if the
    shape is also the same. Segmentation layers (the 1x1(x1) layers that produce the segmentation maps)
    identified by keys ending with '.seg_layers') are not transferred!

    If the pretrained weights were obtained with a training outside nnU-Net and DDP or torch.optimize was used,
    you need to change the keys of the pretrained state_dict. DDP adds a 'module.' prefix and torch.optim adds
    '_orig_mod'. You DO NOT need to worry about this if pretraining was done with nnU-Net as
    nnUNetTrainer.save_checkpoint takes care of that!

    """

    if dist.is_initialized():
        saved_model = torch.load(fname, map_location=torch.device('cuda', dist.get_rank()))
    else:
        saved_model = torch.load(fname)
    if fname.endswith('pth'):
        pretrained_dict = saved_model['network_weights']
    elif fname.endswith('model'):
        pretrained_dict = saved_model['state_dict']

    # print(pretrained_dict.keys())
    if stemid is None:
        skip_strings_in_pretrained = [
            '.seg_layers.', 'seg_outputs',
        ]
    else:
        skip_strings_in_pretrained = [
            '.seg_layers.', 'seg_outputs', 'stem'
        ]
    if dublicate is not None:
        skip_strings_in_pretrained = [
            '.seg_layers.', 'seg_outputs', 'stem'
        ]

    skip_strings_in_pretrained = []
    if isinstance(network, DDP):
        mod = network.module
    else:
        mod = network
    if isinstance(mod, OptimizedModule):
        mod = mod._orig_mod

    model_dict = mod.state_dict()
    # verify that all but the segmentation layers have the same shape
    for key, _ in model_dict.items():
        # print(key)
        if all([i not in key for i in skip_strings_in_pretrained]):
            assert key in pretrained_dict, \
                f"Key {key} is missing in the pretrained model weights. The pretrained weights do not seem to be " \
                f"compatible with your network."

            if not model_dict[key].shape == pretrained_dict[key].shape:
                print('Warning: keys shape do not fit: ', key)
                skip_strings_in_pretrained.append(key)

    for key, _ in model_dict.items():
        if all([i not in key for i in skip_strings_in_pretrained]):
            assert key in pretrained_dict, \
                f"Key {key} is missing in the pretrained model weights. The pretrained weights do not seem to be " \
                f"compatible with your network."
            assert len(skip_strings_in_pretrained) < 7, \
                f"The shape of the parameters of more than 7 keys are not the same. Pretrained model: " \
                f"The pretrained model does not seem to be compatible with your network."

    # fun fact: in principle this allows loading from parameters that do not cover the entire network. For example pretrained
    # encoders. Not supported by this function though (see assertions above)

    # commenting out this abomination of a dict comprehension for preservation in the archives of 'what not to do'
    # pretrained_dict = {'module.' + k if is_ddp else k: v
    #                    for k, v in pretrained_dict.items()
    #                    if (('module.' + k if is_ddp else k) in model_dict) and
    #                    all([i not in k for i in skip_strings_in_pretrained])}

    filtered_pretrained_dict = {k: v for k, v in pretrained_dict.items()
                       if k in model_dict.keys() and all([i not in k for i in skip_strings_in_pretrained])}

    # if stem_id is set we select one stem of certain dataset for
    # print(pretrained_dict.keys())
    if stemid is not None:
        print('You selected stem from dataset %s for pretraining' % stemid)
        for k in model_dict.keys():
            # k 'encoder.stem. ...
            if 'encoder.stem' in k:
                if 'decoder.encoder.stem' in k:
                    filtered_pretrained_dict[k] = pretrained_dict[k[:21] + stemid + '.' + k[21:]]
                else:
                    filtered_pretrained_dict[k] = pretrained_dict[k[:13] + stemid + '.' + k[13:]]

    if dublicate is not None:
        for key, _ in model_dict.items():
            if 'encoder.stem.convs.0.conv.weight' in key or 'encoder.stem.convs.0.all_modules.0.weight' in key:
                if stemid is not None:
                    filtered_pretrained_dict[key] = filtered_pretrained_dict[key].repeat(1, int(dublicate), 1, 1, 1)
                else:
                    filtered_pretrained_dict[key] = pretrained_dict[key].repeat(1, int(dublicate), 1, 1, 1)



    model_dict.update(filtered_pretrained_dict)

    print("################### Loading pretrained weights from file ", fname, '###################')
    if verbose:
        print("Below is the list of overlapping blocks in pretrained model and nnUNet architecture:")
        for key, value in filtered_pretrained_dict.items():
            print(key, 'shape', value.shape)
        print("################### Done ###################")
    mod.load_state_dict(model_dict)



def load_pretrained_weights_encoder(network, fname, verbose=False):
    """
     Transfers all weights between matching keys in state_dicts of the decoder. matching is done by name and we only transfer if the
     shape is also the same.

     If the pretrained weights were obtained with a training outside nnU-Net and DDP or torch.optimize was used,
     you need to change the keys of the pretrained state_dict. DDP adds a 'module.' prefix and torch.optim adds
     '_orig_mod'. You DO NOT need to worry about this if pretraining was done with nnU-Net as
     nnUNetTrainer.save_checkpoint takes care of that!

     """

    if dist.is_initialized():
        saved_model = torch.load(fname, map_location=torch.device('cuda', dist.get_rank()))
    else:
        saved_model = torch.load(fname)
    if fname.endswith('pth'):
        pretrained_dict = saved_model['network_weights']
    elif fname.endswith('model'):
        pretrained_dict = saved_model['state_dict']

    skip_strings_in_pretrained = [
        '.seg_layers.', 'seg_outputs','decoder'
    ]

    if isinstance(network, DDP):
        mod = network.module
    else:
        mod = network
    if isinstance(mod, OptimizedModule):
        mod = mod._orig_mod

    model_dict = mod.state_dict()
    # verify that all but the segmentation layers have the same shape
    for key, _ in model_dict.items():
        if all([i not in key for i in skip_strings_in_pretrained]):
            assert key in pretrained_dict, \
                f"Key {key} is missing in the pretrained model weights. The pretrained weights do not seem to be " \
                f"compatible with your network."

            if not model_dict[key].shape == pretrained_dict[key].shape:
                print('Warning: keys shape do not fit: ', key)
                skip_strings_in_pretrained.append(key)

    for key, _ in model_dict.items():
        if all([i not in key for i in skip_strings_in_pretrained]):
            assert key in pretrained_dict, \
                f"Key {key} is missing in the pretrained model weights. The pretrained weights do not seem to be " \
                f"compatible with your network."
            assert len(skip_strings_in_pretrained) < 7, \
                f"The shape of the parameters of more than 7 keys are not the same. Pretrained model: " \
                f"The pretrained model does not seem to be compatible with your network."

    # fun fact: in principle this allows loading from parameters that do not cover the entire network. For example pretrained
    # encoders. Not supported by this function though (see assertions above)

    # commenting out this abomination of a dict comprehension for preservation in the archives of 'what not to do'
    # pretrained_dict = {'module.' + k if is_ddp else k: v
    #                    for k, v in pretrained_dict.items()
    #                    if (('module.' + k if is_ddp else k) in model_dict) and
    #                    all([i not in k for i in skip_strings_in_pretrained])}

    pretrained_dict = {k: v for k, v in pretrained_dict.items()
                       if k in model_dict.keys() and all([i not in k for i in skip_strings_in_pretrained])}

    model_dict.update(pretrained_dict)

    print("################### Loading pretrained weights from file ", fname, '###################')
    if verbose:
        print("Below is the list of overlapping blocks in pretrained model and nnUNet architecture:")
        for key, value in pretrained_dict.items():
            print(key, 'shape', value.shape)
        print("################### Done ###################")
    mod.load_state_dict(model_dict)
