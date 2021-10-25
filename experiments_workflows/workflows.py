import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from imblearn.under_sampling \
    import (RandomUnderSampler,
            TomekLinks,
            InstanceHardnessThreshold)
from imblearn.over_sampling \
    import (ADASYN,
            RandomOverSampler,
            SMOTE)
from imblearn.combine import SMOTETomek
from experiments_workflows.isolation_forest import IsolationForestUnsupervised

from experiments_workflows.general import prepare_training_set, xy_retrieval
from threshold.optimization import clf_threshold_selection
from experiments_workflows.ll import LayeredLearning


class Workflows:

    def __init__(self,
                 dataset,
                 train_index,
                 test_index,
                 ep_model,
                 resample_size,
                 resample_on_positives):

        training = \
            ep_model.subset_episodes(dataset=dataset,
                                     ind=train_index,
                                     resample_episodes=True,
                                     resample_on_positives=resample_on_positives,
                                     resample_size=resample_size)

        testing = \
            ep_model.subset_episodes(dataset=dataset,
                                     ind=test_index,
                                     resample_episodes=False)

        training_df, train_sub_df, validation_df = prepare_training_set(training)

        self.imputation_model = SimpleImputer(strategy='median')

        self.training = training
        self.training_df = training_df
        self.train_sub_df = train_sub_df
        self.validation_df = validation_df
        self.testing = testing
        self.train_index = train_index
        self.test_index = test_index
        self.ep_model = ep_model
        self.thr = 0

    def standard_classification(self,
                                model=RandomForestClassifier(),
                                resample_distribution: bool = True,
                                resampling_function=SMOTE(),
                                probabilistic_output: bool = False,
                                use_f1: bool = False):

        X_tr, y_tr, _, _ = xy_retrieval(self.training_df, self.ep_model.target_variable)
        X_subtr, y_subtr, _, _ = xy_retrieval(self.train_sub_df, self.ep_model.target_variable)
        X_vld, y_vld, _, _ = xy_retrieval(self.validation_df, self.ep_model.target_variable)

        self.imputation_model.fit(X_subtr)

        print(pd.Series(y_tr).value_counts() / len(y_tr))

        if resample_distribution:
            print('fit res')
            X_tr, y_tr = resampling_function.fit_resample(X_tr, y_tr)
            X_subtr, y_subtr = resampling_function.fit_resample(X_subtr, y_subtr)
            print(pd.Series(y_tr).value_counts() / len(y_tr))

        print('fit thr')
        self.thr = clf_threshold_selection(X_subtr, y_subtr, X_vld, y_vld, model, use_f1)
        print(f'Best threshold is {self.thr}')

        model.fit(X_tr, y_tr)

        y_hat_values, y_hat_prob_values, y_values = {}, {}, {}
        for k in self.testing:
            patient_k = self.testing[k]

            patient_k = patient_k.dropna().reset_index(drop=True)

            X_ts, y_ts, _, _ = xy_retrieval(patient_k, self.ep_model.target_variable)

            print(X_ts.shape)
            if X_ts.shape[0] < 1:
                continue

            X_ts_t = self.imputation_model.transform(X_ts)
            X_ts_t = pd.DataFrame(X_ts_t)
            X_ts_t.columns = X_ts.columns

            if probabilistic_output:
                y_hat_k_p = model.predict_proba(X_ts_t)
                y_hat_k_p = np.array([x[1] for x in y_hat_k_p])
                y_hat_k = (y_hat_k_p > self.thr).astype(int)
            else:
                y_hat_k = model.predict(X_ts_t)
                y_hat_k_p = y_hat_k.copy()

            y_hat_values[k] = y_hat_k
            y_hat_prob_values[k] = y_hat_k_p
            y_values[k] = y_ts.values

        return y_hat_values, y_hat_prob_values, y_values

    def ad_hoc_rule(self):

        target_ah = re.sub('_int$', '_dummy', self.ep_model.target_variable)

        y_hat_values, y_values = {}, {}
        for k in self.testing:
            patient_k = self.testing[k]

            patient_k = patient_k.dropna().reset_index(drop=True)

            X_ts, y_ts, _, _ = xy_retrieval(patient_k, self.ep_model.target_variable)

            if X_ts.shape[0] < 1:
                continue

            y_hat_k = patient_k[target_ah].values

            y_hat_values[k] = y_hat_k
            y_values[k] = y_ts.values

        return y_hat_values, y_values

    def layered_learning(self,
                         model_t1=RandomForestClassifier(),
                         model_t2=RandomForestClassifier(),
                         resample_distribution: bool = True,
                         resampling_function=SMOTE(),
                         probabilistic_output: bool = False,
                         use_f1: bool = False):

        X_tr, y_tr, y_pce_tr, _ = xy_retrieval(self.training_df, self.ep_model.target_variable)

        self.imputation_model.fit(X_tr)

        X_t1_tr, y_t1_tr, X_t2_tr, y_t2_tr = LayeredLearning.formalization(X_tr, y_tr, y_pce_tr)

        best_thr = LayeredLearning.threshold_opt(X_tr=X_tr,
                                                 y_tr=y_tr,
                                                 y_pce_tr=y_pce_tr,
                                                 algo_t1=model_t1,
                                                 algo_t2=model_t2,
                                                 use_f1=use_f1)

        print('best_thr')
        print(best_thr)

        print(pd.Series(y_t1_tr).value_counts() / len(y_t1_tr))
        print(pd.Series(y_t2_tr).value_counts() / len(y_t2_tr))

        if resample_distribution:
            X_t1_tr, y_t1_tr = resampling_function.fit_resample(X_t1_tr, y_t1_tr)
            X_t2_tr, y_t2_tr = resampling_function.fit_resample(X_t2_tr, y_t2_tr)

        model_t1.fit(X_t1_tr, y_t1_tr)
        model_t2.fit(X_t2_tr, y_t2_tr)

        y_hat_values, y_hat_prob_values, y_values = {}, {}, {}
        for k in self.testing:
            patient_k = self.testing[k]
            X_ts, y_ts, _, _ = xy_retrieval(patient_k, self.ep_model.target_variable)

            print(X_ts.shape)
            if X_ts.shape[0] < 1:
                continue

            X_ts_t = self.imputation_model.transform(X_ts)

            if probabilistic_output:
                y_hat_k_p, y_hat_k_p1, y_hat_k_p2 = \
                    LayeredLearning.predict_proba(X_ts_t,
                                                  model_t1=model_t1,
                                                  model_t2=model_t2)

                y_hat_k = np.asarray(y_hat_k_p > best_thr).astype(int)
            else:
                y_hat_k, _, _ = LayeredLearning.predict(X_ts_t, model_t1=model_t1, model_t2=model_t2)
                y_hat_k_p = y_hat_k.copy()

            y_hat_values[k] = y_hat_k
            y_hat_prob_values[k] = y_hat_k_p
            y_values[k] = y_ts.values

        return y_hat_values, y_hat_prob_values, y_values

    def isolation_forest(self,
                         probabilistic_output: bool = False,
                         use_f1: bool = False):

        X_tr, y_tr, _, _ = xy_retrieval(self.training_df, self.ep_model.target_variable)
        X_subtr, y_subtr, _, _ = xy_retrieval(self.train_sub_df, self.ep_model.target_variable)
        X_vld, y_vld, _, _ = xy_retrieval(self.validation_df, self.ep_model.target_variable)

        self.imputation_model.fit(X_subtr)

        print(pd.Series(y_tr).value_counts() / len(y_tr))

        model = IsolationForestUnsupervised()

        print('fit thr')
        self.thr = clf_threshold_selection(X_subtr, y_subtr, X_vld, y_vld, model, use_f1)
        print(f'Best threshold is {self.thr}')

        model.fit(X_tr, y_tr)

        y_hat_values, y_hat_prob_values, y_values = {}, {}, {}
        for k in self.testing:
            patient_k = self.testing[k]

            patient_k = patient_k.dropna().reset_index(drop=True)

            X_ts, y_ts, _, _ = xy_retrieval(patient_k, self.ep_model.target_variable)

            print(X_ts.shape)
            if X_ts.shape[0] < 1:
                continue

            X_ts_t = self.imputation_model.transform(X_ts)
            X_ts_t = pd.DataFrame(X_ts_t)
            X_ts_t.columns = X_ts.columns

            if probabilistic_output:
                y_hat_k_p = np.asarray(model.predict_proba(X_ts_t))
                # y_hat_k_p = np.array([x[1] for x in y_hat_k_p])
                y_hat_k = (y_hat_k_p > self.thr).astype(int)
            else:
                y_hat_k = model.predict(X_ts_t)
                y_hat_k_p = y_hat_k.copy()

            y_hat_values[k] = y_hat_k
            y_hat_prob_values[k] = y_hat_k_p
            y_values[k] = y_ts.values

        return y_hat_values, y_hat_prob_values, y_values
