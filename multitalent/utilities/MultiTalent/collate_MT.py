from typing import List

import numpy as np


def collate_outputs(outputs: List[dict]):
    """
    used to collate default train_step and validation_step outputs. If you want something different then you gotta
    extend this

    we expect outputs to be a list of dictionaries where each of the dict has the same set of keys
    """
    collated = {}
    for k in outputs[0].keys():
        if np.isscalar(outputs[0][k]):
            collated[k] = [o[k] for o in outputs]
        elif isinstance(outputs[0][k], np.ndarray):
            collated[k] = np.vstack([o[k][None] for o in outputs])
        elif isinstance(outputs[0][k], list):
            collated[k] = [item for o in outputs for item in o[k]]
        else:
            raise ValueError(f'Cannot collate input of type {type(outputs[0][k])}. '
                             f'Modify collate_outputs to add this functionality')
    return collated

def collate_outputs_MT(outputs: List[dict], all_ids:list):
    """
    used to collate default train_step and validation_step outputs. If you want something different then you gotta
    extend this

    we expect outputs to be a list of dictionaries where each of the dict has the same set of keys
    """
    collated = {}
    collated['loss'] = []
    for key in all_ids:
        if key not in collated.keys():
            collated[key] = {}
            collated[key]['fp_hard'] = []
            collated[key]['tp_hard'] = []
            collated[key]['fn_hard'] = []
    for entry in outputs:
        for key in entry['fp_hard'].keys():
            collated[key]['fp_hard'].append(entry['fp_hard'][key])
            collated[key]['tp_hard'].append(entry['tp_hard'][key])
            collated[key]['fn_hard'].append(entry['fn_hard'][key])
        collated['loss'].append(entry['loss'])

    return collated