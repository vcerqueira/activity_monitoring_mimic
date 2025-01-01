import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import IsolationForest

from pprint import pprint

import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_auc_score, classification_report
from sklearn.model_selection import KFold, StratifiedKFold
from algorithms.xgb import XGBoostClf
from algorithms.lightgbm import LightGBMClassifier

from data_models.episode import EpisodeModel
from experiments_workflows.general import prepare_training_set, xy_retrieval, compute_eval_metrics
from experiments_workflows.workflows import Workflows
from evaluation.metrics import ActivityMonitoringEvaluation
from utils.files import load_data, save_data

pd.set_option('display.max_columns', 500)

file_path = './mimic_patients_complete.pkl'  # complete db
# file_path = '/Users/vcerqueira/Desktop/mimic_patients_1_300.pkl'  # laptop sample
with open(file_path, 'rb') as fp:
    dataset = pickle.load(fp)

# patients = [*dataset]
# dataset[patients[0]]

TARGET_VARIABLE = 'target_hypotension_int'

ep_model = EpisodeModel(target_variable='target_hypotension_int',
                        min_ep_duration=150,
                        max_event_duration=60,
                        positive_entities_only=False)

has_episode = [int((dataset[k][TARGET_VARIABLE].dropna() > 0).any()) for k in dataset]

cv = StratifiedKFold(n_splits=10, shuffle=True)

patients_ids = [*dataset]

# for train_index, test_index in cv.split(patients_ids):
#     print("TRAIN SIZE:", len(train_index), "TEST SIZE:", len(test_index))
#     print(train_index[:3])
#
#     X_train, X_test = X.values[train_index], X.values[test_index]
#     y_train, y_test = y[train_index], y[test_index]


train_index, test_index = next(cv.split(patients_ids, has_episode))

wf = Workflows(dataset=dataset,
               train_index=train_index[:300],
               test_index=test_index,
               ep_model=ep_model,
               resample_size=30,
               resample_on_positives=True)


y_hat_ah, y_ah = wf.ad_hoc_rule()

am_metrics_ah, cr_ah, amoc_ah = compute_eval_metrics(y_hat_ah, y_ah, None)
pprint(cr_ah)
pprint(am_metrics_ah)

y_hat_if, y_hat_p_if, y_if = wf.isolation_forest(
    probabilistic_output=True,
    use_f1=False)

am_metrics_if, cr_if, amoc_if = compute_eval_metrics(y_hat_if, y_if, y_hat_p_if)
pprint(cr_if)
pprint(am_metrics_if)

y_hat, y_hat_p, y = wf.standard_classification(resample_distribution=True,
                                               probabilistic_output=True,
                                               model=LightGBMClassifier(),
                                               resampling_function=SMOTE(),
                                               use_f1=False)

am_metrics, cr, amoc_clf = compute_eval_metrics(y_hat, y, y_hat_p)
pprint(cr)
pprint(am_metrics)

y_hat_ll, y_hat_p_ll, y_ll = wf.layered_learning(resample_distribution=True,
                                                 probabilistic_output=True,
                                                 model_t1=LightGBMClassifier(),
                                                 model_t2=LightGBMClassifier(),
                                                 resampling_function=SMOTE(),
                                                 use_f1=False)

am_metrics_ll, cr_ll, amoc_ll = compute_eval_metrics(y_hat_ll, y_ll, y_hat_p_ll)
pprint(am_metrics_ll)
pprint(cr_ll)
#
pprint(am_metrics)
pprint(am_metrics_ll)
pprint(am_metrics_if)

amoc_clf['method'] = 'clf'
amoc_if['method'] = 'if'
amoc_ll['method'] = 'll'

df = pd.concat([amoc_clf, amoc_ll, amoc_if], axis=0)
df.to_csv('result_amoc.csv')
