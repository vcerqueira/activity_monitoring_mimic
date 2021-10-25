from typing import Dict
import numpy as np
import pandas as pd


class EpisodeEvaluation:
    """
    Evaluating activity monitoring data_models in a given episode
    Metrics include:
    1 - Anticipation Time;
    2 - Discounted False Positives
    3 - False Alarms per unit of time
    """

    @classmethod
    def anticipation_time(cls,
                          y_hat: np.ndarray,
                          y: np.ndarray):
        """

        :param y_hat: (1d arr) vector of binary predictions
        :param y: (1d arr) true binary values

        :return: Anticipation score which denotes how soon the model detects the event.
        - 1 means that the event was predicted in the (defined) perfect moment
        - 0 means that the event was not detected
        - nan means no event happened at all
        """
        # assert issubclass(y.dtype.type, np.integer)

        if not isinstance(y, np.ndarray):
            y = np.array(y)
            y_hat = np.array(y_hat)

        event_happens = any(y > 0)

        if not event_happens:
            return np.nan

        acceptable_yh_zone = np.where(y > 0)[0]

        yh_on_az = y_hat[acceptable_yh_zone]
        true_alarms = yh_on_az > 0

        if any(true_alarms):
            alarm_ind = np.min(np.where(true_alarms)[0])
            no_anticipation_periods = len(acceptable_yh_zone) - alarm_ind
        else:
            no_anticipation_periods = 0

        anticipation_score = no_anticipation_periods / len(acceptable_yh_zone)

        return anticipation_score

    @staticmethod
    def discounted_false_positives(y_hat: np.ndarray,
                                   y: np.ndarray,
                                   sleep_period: int):
        """
        Discounted False Positive - A metric for false positive but which takes into account
        sequential FP. After a false positive prediction (which counts as 1 discounted FP),
        the predictions are 'turned off' for a sleep_period number of periods.

        :param y_hat: (1d arr) vector of binary predictions
        :param y: (1d arr) true binary values
        :param sleep_period: (int) number of periods to turn off model

        :return: DFP count - the lower the better
        """
        neg_activity = y < 1

        if any(neg_activity):
            y_neg_act = y[y < 1]
            yh_neg_act = y_hat[y < 1]
        else:
            return 0

        false_alarm_count, i = 0, 0
        while i < len(y_neg_act):
            is_false_alarm = yh_neg_act[i] != y_neg_act[i]
            if is_false_alarm:
                false_alarm_count += 1
                i += sleep_period
            else:
                i += 1

        return false_alarm_count

    @classmethod
    def false_alarms_unit(cls, y_hat, y, n_periods):
        """

        :param y_hat: (1d arr) vector of binary predictions
        :param y: (1d arr) true binary values
        :param n_periods: (int) number of periods to average the number of false alarms

        :return: False alarm ratio score
        """

        neg_activity = y < 1
        if not any(neg_activity):
            return 0

        yh_neg_act = y_hat[neg_activity]

        false_alarms_p_unit = \
            np.sum(yh_neg_act) / (len(yh_neg_act) / n_periods)

        return false_alarms_p_unit


class ActivityMonitoringEvaluation:
    """
    Evaluating an activity monitoring model across multiple episodes
    """

    def __init__(self,
                 n_periods: int = 60,
                 sleep_period: int = 60):
        """

        :param n_periods: (int) Number of periods for the false alarm ratio
        :param sleep_period: (int) Number of periods to turn off classifier after a
        positive prediction
        """
        self.n_periods, self.sleep_period = n_periods, sleep_period

        self.at, self.dfp, self.fa = np.array([]), np.array([]), np.array([])

        self.metrics = {}

    def episode_eval(self,
                     y_hat_dict: Dict,
                     y_dict: Dict):
        """
        Evaluating a classifier in a dictionary of episodes

        :param y_hat_dict: (dict) Predictions of the model for each episode
        :param y_dict: True values for each episode

        :return: self, with the following metrics computed for each episode:
        - anticipation time;
        - discounted false positives;
        - false alarm rate
        """
        ep_eval = EpisodeEvaluation()

        for k in y_hat_dict:
            yh = y_hat_dict[k]
            y = y_dict[k]

            self.at = np.append(self.at, ep_eval.anticipation_time(yh, y))
            self.dfp = np.append(self.dfp, ep_eval.discounted_false_positives(yh, y, self.sleep_period))
            self.fa = np.append(self.fa, ep_eval.false_alarms_unit(yh, y, self.n_periods))

    def reduced_precision(self):
        """

        :return: Reduced precision metric
        """

        rp = np.nansum(self.at) / (np.nansum(self.at) + np.sum(self.dfp))

        return rp

    @classmethod
    def amoc_points(cls, y_hat_p, y, n_periods):
        score = pd.DataFrame([])
        for thr in np.arange(0.0, 1.0, 0.005):

            at_score, far_score = [], []
            for k in y_hat_p:
                yh_prob_p = y_hat_p[k]
                y_p = y[k]

                yh_p = np.asarray(yh_prob_p > thr).astype(int)

                at_p = EpisodeEvaluation.anticipation_time(y_hat=yh_p, y=y_p)
                far_p = EpisodeEvaluation.false_alarms_unit(y_hat=yh_p, y=y_p, n_periods=n_periods)

                at_score.append(at_p)
                far_score.append(far_p)

            at_score = np.array(at_score)
            at_no_nan = at_score[~np.isnan(at_score)]
            er = np.sum(at_no_nan > 0) / len(at_no_nan)

            # score = score.append(pd.DataFrame([thr, np.nanmean(at_score), np.nanmean(far_score) / n_periods]).T)
            score = score.append(pd.DataFrame([thr, np.nanmean(at_score), er, np.nanmean(far_score) / n_periods]).T)

        score.columns = ['thr', 'avg_at', 'er', 'avg_far']

        return score

    def reduced_precision_int(self):
        """

        :return: Reduced precision metric
        """

        at_int_vec = np.array([int(x > 0) for x in self.at])

        rp = np.nansum(at_int_vec) / (np.nansum(at_int_vec) + np.sum(self.dfp))

        return rp

    def eval(self,
             y_hat_dict: Dict,
             y_hat_prob_dict: Dict,
             y_dict: Dict):
        """
        Evaluate a classifier across multiple episodes

        :param y_hat_prob_dict:
        :param y_hat_dict: (dict) Predictions of the model for each episode
        :param y_dict: True values for each episode

        :return: dict with multiple metrics
        """

        self.episode_eval(y_hat_dict, y_dict)

        at_no_nan = self.at[~np.isnan(self.at)]

        er = np.sum(at_no_nan > 0) / len(at_no_nan)

        metrics = {'event_recall': er,
                   'reduced_precision': self.reduced_precision(),
                   'reduced_precision_int': self.reduced_precision_int(),
                   'average_at': np.nanmean(self.at),
                   'false_alarm_rate': np.nanmean(self.fa) / self.n_periods,
                   'true_positives': np.nansum(self.at > 0),
                   'false_negatives': np.nansum(self.at == 0),
                   'total_dfp': np.sum(self.dfp),
                   }

        print('Computing AMOC')
        if y_hat_prob_dict is not None:
            amoc_score = \
                self.amoc_points(y_hat_p=y_hat_prob_dict,
                                 y=y_dict,
                                 n_periods=self.n_periods)
        else:
            amoc_score_d = {
                'thr': 0,
                'avg_at': metrics['average_at'],
                'er': er,
                'avg_far': metrics['false_alarm_rate']
            }

            amoc_score = pd.DataFrame(amoc_score_d.values()).T
            amoc_score.columns = amoc_score_d.keys()

        print('F Computing AMOC')

        return metrics, amoc_score
