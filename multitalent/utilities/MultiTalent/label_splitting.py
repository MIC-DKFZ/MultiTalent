import numpy as np


def split_dict_with_background(d, max_num:int=20):
    keys = list(d.keys())
    num_keys = len(keys)

    # Remove "background" key from the keys list
    background = d.pop("background")
    keys.remove("background")

    # Calculate the number of parts
    num_parts = (num_keys + (max_num-1)) // max_num  # Round up

    # Calculate the number of keys per part (excluding "background")
    keys_per_part = num_keys // num_parts

    # Distribute the keys as evenly as possible among the parts
    parts = []
    start_index = 0
    for i in range(num_parts):
        end_index = min(start_index + keys_per_part, num_keys)
        part_keys = keys[start_index:end_index]
        part = {key: d[key] for key in part_keys}
        parts.append(part)
        start_index = end_index

    # Add "background" key to each part
    for part in parts:
        part["background"] = background
    return parts, num_parts


def sample_from_dict(prob_dict, n, last_ids):
    keys = list(prob_dict.keys())
    probs = list(prob_dict.values())
    sampled_keys = []
    while len(sampled_keys) < n:
        sampled_key = np.random.choice(keys, p=probs, size=1, replace=False)
        if sampled_key[0] not in last_ids and sampled_key[0] not in sampled_keys:
            sampled_keys.append(sampled_key[0])
        # sampled_keys.append(sampled_key[0])
    return sampled_keys

if __name__ == '__main__':
    last_ids = []
    for i in range(100):
        dict = {"a":0.01, "b": 0.02, "c": 0.01, "d":0.01, "e":0.01, "f": 0.02, "g": 0.01, "h":0.01, "i":0.9}
        ids = sample_from_dict(dict,2, last_ids)
        for id in ids:
            last_ids.append(id)
        if len(last_ids) == 2* 4:
            last_ids = last_ids[2:]