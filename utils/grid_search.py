from typing import Dict
import numpy as np
import itertools
import pandas as pd


def expand_grid(*iters):
    product = list(itertools.product(*iters))
    return {'Var{}'.format(i + 1): [x[i] for x in product]
            for i in range(len(iters))}


def expand_grid_from_dict(x: Dict) -> pd.DataFrame:
    param_grid = expand_grid(*x.values())
    param_grid = pd.DataFrame(param_grid)
    param_grid.columns = x.keys()

    return param_grid


def parse_config(x: pd.Series) -> Dict:
    """
    todo doc

    :param x:
    :return:
    """
    config = dict(x)

    for key in config:
        try:
            if np.isnan(config[key]):
                config[key] = None
        except TypeError:
            continue

    return config
