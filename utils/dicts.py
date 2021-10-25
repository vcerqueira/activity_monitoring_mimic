from typing import Dict, List
from itertools import compress
import numpy as np


def flatten_dict(x: Dict):
    """ Flat dict on a single level
    :param x: Dict
    :return: Flat dict
    """

    flat_dict = dict()
    for k in x:
        for k2 in x[k]:
            new_k = k + '_PT' + str(k2)
            flat_dict[new_k] = x[k][k2]

    return flat_dict


def dict_subset_by_key(x: Dict, keys: List) -> Dict:
    new_dict = {k: x[k] for k in keys}

    return new_dict


def dict_subset_by_bool(x: Dict, ind: List) -> Dict:
    valid_keys = list(compress(x, ind))

    out = dict_subset_by_key(x, valid_keys)

    return out


def dict_argmax(x):
    x_values = [*x.values()]
    x_best_key = [*x][int(np.argmax(x_values))]

    return x_best_key


