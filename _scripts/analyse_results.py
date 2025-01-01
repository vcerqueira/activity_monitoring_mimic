from pprint import pprint

import pandas as pd
from utils.files import load_data, save_data

from experiments_workflows.general import compute_eval_metrics


# iter_data_results = load_data('kfcv_results_ahe.pkl')
iter_data_results = load_data('kfcv_results_tachypena.pkl')

y_hat_ah, y_ah = {}, {}
y_hat_if, y_if, y_hat_p_if = {}, {}, {}
y_hat_clf, y_clf, y_hat_p_clf = {}, {}, {}
y_hat_ll, y_ll, y_hat_p_ll = {}, {}, {}
for elem in iter_data_results:
    y_hat_ah = {**y_hat_ah, **elem['y_hat_ah']}
    y_ah = {**y_ah, **elem['y_ah']}
    #
    y_hat_if = {**y_hat_if, **elem['y_hat_if']}
    y_if = {**y_if, **elem['y_if']}
    y_hat_p_if = {**y_hat_p_if, **elem['y_hat_p_if']}
    #
    y_hat_clf = {**y_hat_clf, **elem['y_hat']}
    y_clf = {**y_clf, **elem['y']}
    y_hat_p_clf = {**y_hat_p_clf, **elem['y_hat_p']}
    #
    y_hat_ll = {**y_hat_ll, **elem['y_hat_ll']}
    y_ll = {**y_ll, **elem['y_ll']}
    y_hat_p_ll = {**y_hat_p_ll, **elem['y_hat_p_ll']}


ah_metrics, cr_ah, amoc_ah = compute_eval_metrics(y_hat_ah, y_ah, None)
am_metrics_ll, cr_ll, amoc_ll = compute_eval_metrics(y_hat_ll, y_ll, y_hat_p_ll)
am_metrics_clf, cr_clf, amoc_clf = compute_eval_metrics(y_hat_clf, y_clf, y_hat_p_clf)
am_metrics_if, cr_if, amoc_if = compute_eval_metrics(y_hat_if, y_if, y_hat_p_if)

pprint(ah_metrics)
pprint(am_metrics_ll)
pprint(am_metrics_clf)
pprint(am_metrics_if)

amoc_ah['method'] = 'AH'
amoc_clf['method'] = 'CL'
amoc_if['method'] = 'IF'
amoc_ll['method'] = 'LL'

df = pd.concat([amoc_clf, amoc_ll, amoc_if, amoc_ah], axis=0)
df.to_csv('result_amoc_tachypena.csv')