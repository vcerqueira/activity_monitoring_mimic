from typing import Dict
import pandas as pd
import numpy as np

from feature_extraction.statistics_set \
    import (ccf_features,
            wavelet_features,
            STATS_FUN_UNIV)

STATS_FUN_DUMMY = \
    dict(
        mean=np.nanmean
    )


def episode_dynamics_benchmark(X: pd.DataFrame,
                               func_list: Dict = STATS_FUN_UNIV):
    col_names = X.columns.to_list()

    feature_set = []
    for feature_func in func_list:
        out = X.apply(func_list[feature_func])
        out.index = [x + '_' + feature_func for x in col_names]

        feature_set.append(out)

    feature_set.append(ccf_features(X))
    feature_set.append(wavelet_features(X))

    feature_set = pd.concat(feature_set)

    return feature_set


def episode_dynamics_dummy(X: pd.DataFrame,
                           func_list: Dict = STATS_FUN_DUMMY):
    col_names = X.columns.to_list()

    feature_set = []
    for feature_func in func_list:
        out = X.apply(func_list[feature_func])
        out.index = [x + '_' + feature_func for x in col_names]

        feature_set.append(out)

    feature_set.append(ccf_features(X))

    feature_set = pd.concat(feature_set)

    return feature_set