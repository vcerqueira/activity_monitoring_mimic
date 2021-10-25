import copy
import numpy as np
import pandas as pd
from typing import Tuple, Union


class PanelData:

    def __init__(self,
                 prototype: pd.DataFrame,
                 periods: Tuple[int, int, int]):
        """ PanelData class constructor

        :param prototype: An entity (as pd.DataFrame) that represents the structure of each
        element/entity in the panel
        :param periods: Tuple with three integers, which represents the sizes of each period:
        the observation period, the warning period, and the target period:
         - The observation period is the period available for computing features;
         - The target period denotes the interval for computing the target variable;
         - The warning period denotes the necessary interval for a prediction to be useful.

         An entity denotes a element from the panel _data
         An episode denotes a contiguous snapshot of observation+warning+target for a given
         entity in the panel
        """

        assert isinstance(prototype, pd.DataFrame)

        self.columns = prototype.columns
        self.dtypes = prototype.dtypes
        self.p_observation, self.p_warning, self.p_target = periods
        self.episode_len = self.p_observation + self.p_warning + self.p_target

    def check_entity(self, X: pd.DataFrame):
        """ Checking if input entity has the same structure as self.prototype

        :param X: Input episode as pd.DataFrame
        :return: Boolean
        """

        assert all(X.columns == self.columns)
        assert all(X.dtypes == self.dtypes)
        assert X.shape[0] >= self.episode_len

    def threshold_target(self,
                         X_: pd.DataFrame,
                         target_variable: str,
                         value_threshold: Union[float, int],
                         ratio_threshold: float,
                         get_dummy: bool,
                         above_threshold: bool,
                         get_ratio: bool = False):
        """ Creating targets using a threshold-based approach

        :param X_: Episode as pd.DataFrame with the same structure as the prototype
        :param target_variable: String describing the target variable
        :param value_threshold: Value above/below which the target is positive
        :param ratio_threshold: Threshold of values that need to be on the value_threshold for
        the target to be positive
        :param get_dummy: whether to return variable that only looks at the num threshold
        :param above_threshold: Whether the positive target is above or below the value_threshold
        :param get_ratio: Output the ratio of values meeting the condition
        :return: A target variable for the episode
        """

        X = copy.deepcopy(X_)

        self.check_entity(X)

        x = X[target_variable]

        if above_threshold:
            value_condition = x > value_threshold
        else:
            value_condition = x < value_threshold

        if get_dummy:
            return value_condition.astype(int)

        if get_ratio:
            out = value_condition.rolling(self.p_target).sum() / self.p_target
            out = out.values
        else:
            ratio_condition = \
                value_condition.rolling(self.p_target).sum() / self.p_target > ratio_threshold

            out = ratio_condition.values.astype(int)

        return out

    def spot_threshold_target(self,
                              X_: pd.DataFrame,
                              target_variable: str,
                              value_threshold: Union[float, int],
                              ratio_threshold: float,
                              above_threshold: bool,
                              get_ratio: bool = False):
        """ Creating targets using a threshold-based approach

        :param X_: Episode as pd.DataFrame with the same structure as the prototype
        :param target_variable: String describing the target variable
        :param value_threshold: Value above/below which the target is positive
        :param ratio_threshold: Threshold of values that need to be on the value_threshold for
        the target to be positive
        :param above_threshold: Whether the positive target is above or below the value_threshold
        :param get_ratio: Output the ratio of values meeting the condition
        :return: A target variable for the episode
        """

        X = copy.deepcopy(X_)

        NA_THRESHOLD_RATIO = 0.5

        self.check_entity(X)

        X = X[-self.p_target:]

        x = X[target_variable]

        assert isinstance(x, pd.Series)

        if x.isna().sum() / len(x) > NA_THRESHOLD_RATIO:
            return np.nan

        if above_threshold:
            value_condition = x > value_threshold
        else:
            value_condition = x < value_threshold

        if get_ratio:
            out = value_condition.sum() / self.p_target
        else:
            ratio_condition = \
                value_condition.sum() / self.p_target > ratio_threshold

            out = int(ratio_condition)

        return out

    def spot_threshold_dummy(self,
                             X_: pd.DataFrame,
                             target_variable: str,
                             value_threshold: Union[float, int],
                             ratio_threshold,
                             above_threshold: bool):
        """ Creating targets using a threshold-based approach

        :param X_: Episode as pd.DataFrame with the same structure as the prototype
        :param target_variable: String describing the target variable
        :param value_threshold: Value above/below which the target is positive
        :param above_threshold: Whether the positive target is above or below the value_threshold
        :param get_ratio: Output the ratio of values meeting the condition
        :return: A target variable for the episode
        """
        X = copy.deepcopy(X_)

        NA_THRESHOLD_RATIO = 0.5

        self.check_entity(X)

        #print('Shape before')
        #print(X.shape)
        X = X[:self.p_observation]
        #print('Shape after')
        #print(X.shape)

        x = X[target_variable]

        assert isinstance(x, pd.Series)

        if x.isna().sum() / len(x) > NA_THRESHOLD_RATIO:
            return np.nan

        if above_threshold:
            value_condition = x > value_threshold
        else:
            value_condition = x < value_threshold

        value_condition = np.asarray(value_condition).astype(int)

        out = value_condition[-1]

        return out

    def data_points_predictors(self,
                               episode: pd.DataFrame,
                               predictors_fun):
        """ Retrieving the attributes of the episode from the observation period
        :param episode: episode from an entity, containing at least
        the pre-specified observation period
        :param predictors_fun: Function used for retrieving the attributes from the observation
        period

        :return: attributes for the respective episode
        """

        if isinstance(episode, np.ndarray):
            episode = pd.DataFrame(episode, columns=self.columns)

        assert episode.shape[0] == self.episode_len

        observation_period = episode[:self.p_observation]

        if self.check_episode_is_valid(observation_period):

            if self.check_episode_missing_data(observation_period):
                avg_values = observation_period.mean()
                observation_period = observation_period.fillna(avg_values)

            ep_predictors = predictors_fun(observation_period)
            episode_is_valid = True
        else:
            aux_op = np.random.random(observation_period.shape)
            aux_op = pd.DataFrame(aux_op[:30], columns=observation_period.columns)
            aux_attrs = predictors_fun(aux_op)
            ep_predictors = pd.Series(dict.fromkeys(aux_attrs.index, np.nan))
            episode_is_valid = False

        return ep_predictors, episode_is_valid

    def create_instances(self,
                         entity: pd.DataFrame,
                         predictors_fun,
                         **kwargs):
        """ Offline process for creating instances
        :param entity: Entity _data as a pd.DataFrame
        :param predictors_fun: Function for computing features
        :kwargs: keyword arguments for target function
        :return: a pd.DataFrame
        """

        # y_entity = self.threshold_target(entity, **kwargs)

        X, y = [], []
        for i in np.arange(self.episode_len, entity.shape[0] + 1):
            indices = np.arange(i - self.episode_len, i)

            episode = copy.deepcopy(entity.iloc[indices, :])
            X_i, episode_is_valid = self.data_points_predictors(episode,
                                                                predictors_fun=predictors_fun)

            if not episode_is_valid:
                X.append(X_i)
                y.append(np.nan)
            else:
                # y_episode = y_entity[indices]
                # y_i = y_episode[-self.p_target:][-1]
                y_i = self.spot_threshold_target(X_=episode, **kwargs)

                X.append(X_i)
                y.append(y_i)

        X = pd.concat(X, axis=1).T

        return X, y

    def create_instances_all(self,
                             entity: pd.DataFrame,
                             predictors_fun,
                             target_specs):
        """ Offline process for creating instances
        :param entity: Entity _data as a pd.DataFrame
        :param predictors_fun: Function for computing features
        :param target_specs: --
        :return: a pd.DataFrame
        """

        # y_entity = self.threshold_target(entity, **kwargs)

        X, Y = [], []
        for i in np.arange(self.episode_len, entity.shape[0] + 1):

            y = {}

            indices = np.arange(i - self.episode_len, i)

            episode = copy.deepcopy(entity.iloc[indices, :])
            X_i, episode_is_valid = self.data_points_predictors(episode,
                                                                predictors_fun=predictors_fun)

            if not episode_is_valid:
                X.append(X_i)

                for clinical_event in target_specs:
                    #
                    y[f'target_{clinical_event}_int'] = np.nan
                    y[f'target_{clinical_event}_num'] = np.nan
                    y[f'target_{clinical_event}_dummy'] = np.nan
                    y[f'target_{clinical_event}_pce'] = np.nan

                Y.append(pd.Series(y))

            else:
                for clinical_event in target_specs:
                    kw_args = target_specs[clinical_event]
                    #
                    kw_args_pce = kw_args.copy()
                    kw_args_pce.pop('ratio_threshold')
                    #
                    y[f'target_{clinical_event}_int'] = self.spot_threshold_target(X_=episode, **kw_args, get_ratio=False)
                    y[f'target_{clinical_event}_num'] = self.spot_threshold_target(X_=episode, **kw_args, get_ratio=True)
                    y[f'target_{clinical_event}_dummy'] = self.spot_threshold_dummy(X_=episode, **kw_args)
                    y[f'target_{clinical_event}_pce'] = \
                        self.spot_threshold_target(X_=episode, **kw_args_pce, ratio_threshold=0.45)

                X.append(X_i)
                Y.append(pd.Series(y))

        X = pd.concat(X, axis=1).T
        Y = pd.concat(Y, axis=1).T

        return X, Y

    @staticmethod
    def check_episode_is_valid(episode: pd.DataFrame):

        COL_NAMES = ['HR', 'SBP', 'DBP', 'MAP']

        def signal_is_valid(signal_values):
            is_valid = [10 < x < 200 for x in signal_values]

            out = np.sum(is_valid) / len(is_valid) >= 0.8

            return out

        validity_check = episode.apply(signal_is_valid)

        out = all(validity_check[COL_NAMES])

        return out

    @staticmethod
    def check_episode_missing_data(episode: pd.DataFrame):

        na_observations = episode.isna()

        any_na = any(na_observations.sum() > 0)

        return any_na
