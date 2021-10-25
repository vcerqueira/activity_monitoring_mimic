import numpy as np


def na_ratio(x: np.ndarray):
    """ Compute ratio of NA of a np.ndarray

    :param x: array
    :return: ratio of NAs
    """
    ratio = np.sum(np.isnan(x)) / len(x)

    return ratio


def impute_with_median(x: np.ndarray):
    """ Impute the NAs of a np.ndarray with the median

    :param x: array as np.ndarray
    :return: imputed array
    """
    assert isinstance(x, np.ndarray)
    assert np.any(np.isnan(x))

    x[np.isnan(x)] = np.nanmedian(x)

    return x
