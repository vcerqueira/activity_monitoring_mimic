import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from threshold.optimization import optimize_threshold_recalls


class LayeredLearning:

    @classmethod
    def formalization(cls, X: pd.DataFrame, y: np.ndarray, y_pce: np.ndarray):
        """

        :param X:
        :param y:
        :param y_pce:
        :return:
        """
        if isinstance(y, pd.Series):
            y = y.values
        if isinstance(y_pce, pd.Series):
            y_pce = y_pce.values

        X_t1 = X.copy()
        y_t1 = y_pce.copy()

        pce_ind = np.where(y_pce > 0)[0]

        X_t2 = X.iloc[pce_ind, :].reset_index(drop=True)
        y_t2 = y[pce_ind]

        return X_t1, y_t1, X_t2, y_t2

    @classmethod
    def threshold_opt(cls,
                      X_tr: pd.DataFrame,
                      y_tr: pd.Series,
                      y_pce_tr: pd.Series,
                      algo_t1,
                      algo_t2, use_f1):

        if not isinstance(y_tr, np.ndarray):
            if isinstance(y_tr, pd.Series):
                y_tr = y_tr.values
            else:
                y_tr = np.asarray(y_tr)

        seq_tr = np.arange(len(y_tr))
        tr_ind, vl_ind = train_test_split(seq_tr,
                                          test_size=0.2,
                                          stratify=y_tr)

        X_intr = X_tr.iloc[tr_ind, :].reset_index(drop=True)
        X_vl = X_tr.iloc[vl_ind, :].reset_index(drop=True)

        y_intr = y_tr[tr_ind]
        y_vl = y_tr[vl_ind]
        y_pce_intr = y_pce_tr[tr_ind]

        model_t1, model_t2 = cls.modelling(X_intr, y_intr, y_pce_intr, algo_t1, algo_t2)

        yh_prob, _, _ = cls.predict_proba(X_vl, model_t1, model_t2)

        best_thr = optimize_threshold_recalls(yh_prob, y_vl, use_f1)

        return best_thr

    @classmethod
    def modelling(cls, X, y, y_pce, model_t1, model_t2):
        X_t1, y_t1, X_t2, y_t2 = cls.formalization(X=X, y=y, y_pce=y_pce)

        model_t1.fit(X_t1, y_t1)
        model_t2.fit(X_t2, y_t2)

        return model_t1, model_t2

    @classmethod
    def predict_proba(cls, X: pd.DataFrame, model_t1, model_t2):

        yh_prob_t1 = model_t1.predict_proba(X)
        yh_prob_t1 = np.asarray([x[1] for x in yh_prob_t1])

        yh_prob_t2 = model_t2.predict_proba(X)
        yh_prob_t2 = np.asarray([x[1] for x in yh_prob_t2])

        yh_prob = yh_prob_t1 * yh_prob_t2

        return yh_prob, yh_prob_t1, yh_prob_t2

    @classmethod
    def predict(cls, X: pd.DataFrame, model_t1, model_t2):

        yh_t1 = model_t1.predict(X)

        yh_t2 = model_t2.predict(X)

        yh = yh_t1 * yh_t2

        return yh, yh_t1, yh_t2
