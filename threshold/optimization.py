import numpy as np
import pandas as pd

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from utils.dicts import dict_argmax


def optimize_threshold_recalls(y_hat_prob: np.ndarray,
                               y: np.ndarray,
                               use_f1: bool = False):
    score = dict()
    for thr in np.arange(0.01, .99, 0.005):
        yh = (y_hat_prob > thr).astype(int)
        #
        result = classification_report(y, yh, output_dict=True)
        #
        try:
            specificity = result['0']['recall']
            recall = result['1']['recall']
        except KeyError:
            specificity, recall = 0, 0
        #
        if not use_f1:
            score[thr] = (specificity + recall) / 2
        else:
            score[thr] = result['macro avg']['f1-score']
    #
    best_thr = dict_argmax(score)
    #
    return best_thr


def clf_threshold_selection(X_tr: pd.DataFrame,
                            y_tr: pd.Series,
                            X_vl: pd.DataFrame,
                            y_vl: pd.Series,
                            model,
                            use_f1: bool):
    assert isinstance(X_tr, pd.DataFrame)
    assert isinstance(y_tr, pd.Series)

    model.fit(X_tr, y_tr)
    yh_prob = model.predict_proba(X_vl)
    print(yh_prob[0])
    if len(yh_prob[0].shape) > 0:
        yh_prob = np.asarray([x[1] for x in yh_prob])

    best_thr = optimize_threshold_recalls(yh_prob, y_vl.values, use_f1)

    return best_thr
