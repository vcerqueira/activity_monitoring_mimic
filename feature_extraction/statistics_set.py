import numpy as np
import pandas as pd
from scipy import stats
from numpy.linalg import LinAlgError
from scipy.stats import linregress
from scipy import signal
import pywt
import itertools


def norm(x: np.ndarray) -> float:
    """ Norm of vector

    :param x: 1-d numeric vector
    :return: numeric scalar
    """
    try:
        out = np.linalg.norm(x)
    except LinAlgError:
        out = np.nan

    return out


def _relative_energy_wavelets(x, n_levels=5):
    """ Relative energy of wavelets
    todo doc

    :param x:
    :param n_levels:
    :return:
    """
    wave_dec = pywt.wavedec(x, 'db8', level=n_levels)

    a_signal = wave_dec[0]
    a_energy = norm(a_signal)

    d_signals = wave_dec[1:]
    d_energies = [norm(x) for x in d_signals]

    energy_vec = [a_energy] + d_energies

    tot_energy = np.sum(energy_vec)

    relative_energy = energy_vec / tot_energy

    re_names = ['a_re'] + ['re' + str(i) for i in range(1, n_levels + 1)]

    out = pd.Series(relative_energy, index=re_names)

    return out


def wavelet_features(X: pd.DataFrame):
    """
    todo doc
    :param X:
    :return:
    """
    out = X.apply(_relative_energy_wavelets)

    out_wide = []
    for col in out:
        out_col = out[col]
        out_col.index = [x + '_' + col for x in out_col.index]

        out_wide.append(out_col)

    wavelet_rel_energy = pd.concat(out_wide)

    return wavelet_rel_energy


def ccf_features(X: pd.DataFrame):
    """
    todo doc
    :param X:
    :return:
    """
    col_comb = itertools.combinations(X.columns, 2)

    feature_set = []
    feature_names = []
    for v1, v2 in col_comb:
        comb_features = signal.correlate(X[v1].values,
                                         X[v2].values)[0]

        feature_names.append(v1 + '_' + v2 + '_ccf')
        feature_set.append(comb_features)

    out = pd.Series(feature_set, index=feature_names)

    return out


def line_slope(x: np.ndarray) -> float:
    """
    todo doc
    :param x:
    :return:
    """
    lm = linregress(x, list(range(len(x))))

    slope = lm[0]

    return slope


STATS_FUN_UNIV = \
    dict(
        mean=np.nanmean,
        median=np.nanmedian,
        max=np.nanmax,
        sdev=np.nanstd,
        var=np.nanvar,
        skewness=stats.skew,
        kurtosis=stats.kurtosis,
        iqr=stats.iqr,
        slope=line_slope
    )
