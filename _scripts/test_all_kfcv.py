import pickle
from imblearn.over_sampling import SMOTE

from pprint import pprint

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from algorithms.xgb import XGBoostClf
from algorithms.lightgbm import LightGBMClassifier
from utils.files import load_data, save_data

from data_models.episode import EpisodeModel
from experiments_workflows.general import compute_eval_metrics
from experiments_workflows.workflows import Workflows

pd.set_option('display.max_columns', 500)

# file_path = './mimic_patients_complete.pkl'
file_path = './mimic_patients_0_300.pkl'
with open(file_path, 'rb') as fp:
    dataset = pickle.load(fp)

TARGET_VARIABLE = 'target_hypotension_int'
# TARGET_VARIABLE = 'target_tachycardia_int'
# TARGET_VARIABLE = 'target_hypertension_int'
# TARGET_VARIABLE = 'target_bradycardia_int'
# TARGET_VARIABLE = 'target_tachypena_int'
# TARGET_VARIABLE = 'target_bradypena_int'
# TARGET_VARIABLE = 'target_hypoxia_int'

ep_model = EpisodeModel(target_variable=TARGET_VARIABLE,
                        min_ep_duration=150,
                        max_event_duration=60,
                        positive_entities_only=False)

has_episode = [int((dataset[k][TARGET_VARIABLE].dropna() > 0).any()) for k in dataset]

cv = StratifiedKFold(n_splits=10, shuffle=True)

patients_ids = [*dataset]

iter_data_results = []
for train_index, test_index in cv.split(patients_ids, has_episode):
    print("TRAIN SIZE:", len(train_index), "TEST SIZE:", len(test_index))
    #
    wf = Workflows(dataset=dataset,
                   train_index=train_index,
                   test_index=test_index,
                   ep_model=ep_model,
                   resample_size=30,
                   resample_on_positives=True)
    print('running ad hoc')
    y_hat_ah, y_ah = wf.ad_hoc_rule()
    print('running ad hoc metrics')
    am_metrics_ah, cr_ah, amoc_ah = compute_eval_metrics(y_hat_ah, y_ah, None)
    pprint(am_metrics_ah)
    print('running if')
    y_hat_if, y_hat_p_if, y_if = wf.isolation_forest(
        probabilistic_output=True,
        use_f1=False)
    print('running if metrics')
    am_metrics_if, cr_if, amoc_if = compute_eval_metrics(y_hat_if, y_if, y_hat_p_if)
    pprint(am_metrics_if)
    print('running std  clf')
    y_hat, y_hat_p, y = wf.standard_classification(resample_distribution=True,
                                                   probabilistic_output=True,
                                                   model=LightGBMClassifier(),
                                                   resampling_function=SMOTE(),
                                                   use_f1=False)
    print('running std clf metrics')
    am_metrics, cr, amoc_clf = compute_eval_metrics(y_hat, y, y_hat_p)
    pprint(am_metrics)
    print('running ll')
    y_hat_ll, y_hat_p_ll, y_ll = wf.layered_learning(resample_distribution=True,
                                                     probabilistic_output=True,
                                                     model_t1=LightGBMClassifier(),
                                                     model_t2=LightGBMClassifier(),
                                                     resampling_function=SMOTE(),
                                                     use_f1=False)
    print('running ll metrics')
    am_metrics_ll, cr_ll, amoc_ll = compute_eval_metrics(y_hat_ll, y_ll, y_hat_p_ll)
    pprint(am_metrics_ll)
    pprint(cr_ll)
    #
    #
    iter_results = {
        'y_hat_ah': y_hat_ah,
        'y_ah': y_ah,
        'am_metrics_ah': am_metrics_ah,
        'cr_ah': cr_ah,
        'amoc_ah': amoc_ah,
        'y_hat_if': y_hat_if,
        'y_hat_p_if': y_hat_p_if,
        'y_if': y_if,
        'am_metrics_if': am_metrics_if,
        'cr_if': cr_if,
        'amoc_if': amoc_if,
        'y_hat': y_hat,
        'y_hat_p': y_hat_p,
        'y': y,
        'am_metrics': am_metrics,
        'cr': cr,
        'amoc_clf': amoc_clf,
        'y_hat_ll': y_hat_ll,
        'y_hat_p_ll': y_hat_p_ll,
        'y_ll': y_ll,
        'am_metrics_ll': am_metrics_ll,
        'cr_ll': cr_ll,
        'amoc_ll': amoc_ll,
    }
    iter_data_results.append(iter_results)
    save_data(iter_data_results, 'kfcv_results_tachypena22.pkl')

save_data(iter_data_results, 'kfcv_results_tachypena22.pkl')




#
pprint(am_metrics)
pprint(am_metrics_ll)
pprint(am_metrics_if)

amoc_clf['method'] = 'clf'
amoc_if['method'] = 'if'
amoc_ll['method'] = 'll'

df = pd.concat([amoc_clf, amoc_ll, amoc_if], axis=0)
df.to_csv('result_amoc.csv')
