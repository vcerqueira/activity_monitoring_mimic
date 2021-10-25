import numpy as np
import lightgbm as lgbm
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from sklearn.base import BaseEstimator, ClassifierMixin

from utils.grid_search import expand_grid_from_dict, parse_config
from utils.dicts import dict_argmax

parameter_search_model = \
    dict(num_leaves=[15, 31],
         max_depth=[-1, 10],
         lambda_l1=[0.01, 0.1, 0.5, 1],
         lambda_l2=[0.01, 0.1, 0.5, 1],
         feature_fraction=[0.75, 1],
         bagging_fraction=[1, 0.8],
         bagging_freq=[1, 3],
         min_child_samples=[20, 50, 100])

N_JOBS = 2


class LightGBMClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self,
                 boosting_type: str = "gbdt",
                 metric: str = "logloss",
                 early_stopping_rounds: int = 10,
                 num_boost_round: int = 1000,
                 learning_rate=0.2,
                 bagging_fraction=0.8,
                 bagging_freq=1,
                 colsample_bytree=0.85,
                 colsample_bynode=0.85,
                 lambda_l1=0.5,
                 lambda_l2=0.5,
                 num_leaves=31,
                 max_depth=-1,
                 feature_fraction=1,
                 min_child_samples=20,
                 verbose_eval: int = 0,
                 n_iter=25):
        """
        Implementation of the lightgbm method for classification tasks.
        Includes an optimization procedure based on random search.

        :param boosting_type: Algorithm parameter: Please check
        https://lightgbm.readthedocs.io/en/latest/Parameters.html for more information on these
        :param metric: Algorithm-specific parameter
        :param early_stopping_rounds: Algorithm-specific parameter
        :param num_boost_round: Algorithm-specific parameter
        :param learning_rate: Algorithm-specific parameter
        :param bagging_fraction: Algorithm-specific parameter
        :param bagging_freq: Algorithm-specific parameter
        :param colsample_bytree: Algorithm-specific parameter
        :param colsample_bynode: Algorithm-specific parameter
        :param lambda_l1: Algorithm-specific parameter
        :param lambda_l2: Algorithm-specific parameter
        :param num_leaves: Algorithm-specific parameter
        :param max_depth: Algorithm-specific parameter
        :param feature_fraction: Algorithm-specific parameter
        :param min_child_samples: Algorithm-specific parameter
        :param verbose_eval: Algorithm-specific parameter

        :param n_iter Maximum number of random search iterations to find
        the best model
        """
        self.early_stopping_rounds = early_stopping_rounds
        self.num_boost_round = num_boost_round
        self.verbose_eval = verbose_eval
        self.colsample_bytree = colsample_bytree
        self.colsample_bynode = colsample_bynode
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.feature_fraction = feature_fraction
        self.min_child_samples = min_child_samples
        self.learning_rate = learning_rate
        self.bagging_fraction = bagging_fraction
        self.bagging_freq = bagging_freq
        self.n_iter = n_iter
        self.boosting_type = boosting_type

        self.params = \
            dict(objective="binary",
                 metric=metric,
                 verbosity=-1,
                 n_jobs=1,
                 nthreads=1,
                 boosting_type=boosting_type,
                 colsample_bytree=self.colsample_bytree,
                 colsample_bynode=self.colsample_bynode,
                 lambda_l1=self.lambda_l1,
                 lambda_l2=self.lambda_l2,
                 num_leaves=self.num_leaves,
                 max_depth=self.max_depth,
                 feature_fraction=self.feature_fraction,
                 min_child_samples=self.min_child_samples,
                 learning_rate=self.learning_rate,
                 bagging_fraction=self.bagging_fraction,
                 bagging_freq=self.bagging_freq)

        self.model = None
        self.best_iteration = None
        self.best_params = None
        self.test_size = 0.2

    def fit(self, X, y=None):

        X_tr, X_vl, y_tr, y_vl = \
            train_test_split(X, y,
                             test_size=self.test_size,
                             shuffle=True,
                             stratify=y)

        param_df = expand_grid_from_dict(parameter_search_model)

        base_params = \
            dict(objective=self.params['objective'],
                 metric=self.params['metric'],
                 n_jobs=1,
                 nthreads=1,
                 boosting_type='gbdt',
                 early_stopping_rounds=self.early_stopping_rounds,
                 num_boost_round=self.num_boost_round
                 )

        config_id_hist = []
        score_hist = []
        for i in range(self.n_iter):

            config_id = np.random.choice(param_df.shape[0], size=1)[0]
            config_id_hist.append(config_id)

            config = parse_config(param_df.iloc[config_id, :])

            int_cols = ['num_leaves', 'max_depth',
                        'bagging_freq', 'min_child_samples']
            for par in int_cols:
                config[par] = int(config[par])

            params_all = {**base_params, **config}

            model = lgbm.LGBMClassifier(**params_all)

            model.fit(X_tr, y_tr,
                      eval_set=[(X_vl, y_vl)],
                      eval_metric=base_params['metric'])

            y_hat_vl = model.predict(X_vl,
                                     num_iteration=model.best_iteration_)

            score = cohen_kappa_score(y_vl, y_hat_vl)
            score_hist.append(score)

        scores = dict(zip(config_id_hist, score_hist))
        best_config_id = dict_argmax(scores)
        config = parse_config(param_df.iloc[best_config_id, :])

        int_cols = ['num_leaves', 'max_depth',
                    'bagging_freq', 'min_child_samples']
        for par in int_cols:
            config[par] = int(config[par])

        params_all = {**base_params, **config}
        self.params = params_all
        self.model = lgbm.LGBMClassifier(**self.params)

        self.model.fit(X_tr, y_tr,
                       eval_set=[(X_vl, y_vl)],
                       eval_metric=base_params['metric'])
        self.best_iteration = self.model.best_iteration_

        return self

    def predict(self, X):
        y_hat = self.model.predict(X, num_iteration=self.best_iteration)

        return y_hat

    def predict_proba(self, X):
        y_hat = self.model.predict_proba(X, num_iteration=self.best_iteration)

        return y_hat
