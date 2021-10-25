from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_auc_score, classification_report

from evaluation.metrics import ActivityMonitoringEvaluation


def prepare_training_set(train_episodes,
                         drop_na: bool = True):
    """
    :param train_episodes:
    :param drop_na:
    :param split_validation:
    :return:
    """
    train = pd.concat(train_episodes)

    train_patients_ids = [*train_episodes]

    sub_train_idx, validation_idx = train_test_split(train_patients_ids,
                                                     shuffle=True,
                                                     train_size=0.9)

    train_sub_dict = {k: train_episodes[k] for k in sub_train_idx}
    validation_dict = {k: train_episodes[k] for k in validation_idx}

    train_sub = pd.concat(train_sub_dict)
    validation = pd.concat(validation_dict)

    if drop_na:
        train = train.dropna().reset_index(drop=True)
        train_sub = train_sub.dropna().reset_index(drop=True)
        validation = validation.dropna().reset_index(drop=True)

    return train, train_sub, validation


def xy_retrieval(data: pd.DataFrame, target_variable):
    clinical_event = target_variable.split('_')[1]
    #
    targets = dict(y='target_' + clinical_event + '_int',
                   y_pce='target_' + clinical_event + '_pce',
                   y_num='target_' + clinical_event + '_num',
                   y_dummy='target_' + clinical_event + '_dummy')
    #
    target_col_names = \
        [col for col in data.columns
         if col.startswith('target')]
    #
    X = data.drop(target_col_names, axis=1)
    #
    y = data[targets['y']]
    y_pce = data[targets['y_pce']]
    y_num = data[targets['y_num']]
    #
    return X, y, y_pce, y_num


def compute_eval_metrics(y_hat, y, y_hat_prob):
    y_ct = np.concatenate([*y.values()])
    y_hat_ct = np.concatenate([*y_hat.values()])

    is_na = np.isnan(y_ct)

    y_ct = y_ct[~is_na].astype(int)
    y_hat_ct = y_hat_ct[~is_na].astype(int)

    cr = classification_report(y_true=y_ct, y_pred=y_hat_ct, output_dict=True)
    cr['AUC'] = roc_auc_score(y_ct, y_hat_ct)

    eval_am = ActivityMonitoringEvaluation()

    results, amoc = eval_am.eval(y_hat_dict=y_hat, y_dict=y, y_hat_prob_dict=y_hat_prob)

    return results, cr, amoc
