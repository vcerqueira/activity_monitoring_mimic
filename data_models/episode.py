import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from utils.dicts \
    import (flatten_dict,
            dict_subset_by_key)
from utils.data_frames import split_dataframe
from utils.rleid import rleid


class EpisodeModel:

    def __init__(self,
                 target_variable: str,
                 min_ep_duration: int,
                 max_event_duration: int,
                 positive_entities_only: bool):
        """
        Methods for handling episodes.

        :param target_variable: (str) string denoting the target variable
        :param min_ep_duration: (int) minimum number of instances necessary for an episode to be
        valid
        :param max_event_duration: (int) maximum number of positive instances
        :param positive_entities_only: (bool) whether or not to retrieve only episodes with
        positive events
        """

        self.target_variable = target_variable
        self.min_ep_duration = min_ep_duration
        self.max_event_duration = max_event_duration
        self.positive_entities_only = positive_entities_only

    def split_episodes_in_db(self, episode_dict: Dict):
        """
        Splitting the episodes in a dictionary database
        :param episode_dict: (dict) Dictionary of episodes (pd.DF's)

        :return: dictionary with split episodes
        """
        split_episodes = dict()
        for k in episode_dict:
            out = \
                self.episode_split(episode=episode_dict[k],
                                   target_variable=self.target_variable,
                                   max_event_duration=self.max_event_duration,
                                   min_ep_duration=self.min_ep_duration)

            if len(out) > 0:
                if len(out) > 1:
                    split_episodes[k] = out
                else:
                    if not self.positive_entities_only:
                        split_episodes[k] = out

        split_episodes = flatten_dict(split_episodes)

        return split_episodes

    @staticmethod
    def episode_split(episode: pd.DataFrame,
                      target_variable: str,
                      min_ep_duration: int,
                      max_event_duration: int):
        """
        Splitting an episode by event
        :param episode: (pd.DF) episode data
        :param target_variable: (str) target variable
        :param min_ep_duration: (int) minimum number of instance necessary to consider the (
        partial) episode
        :param max_event_duration: (int) maximum number of positive instances to consider the (
        partial) episode

        :return: (dict) dict with split intra episodes
        """
        target_values = episode[target_variable].values

        # rleid on episode target
        encoding_target = rleid(target_values)

        # splitting df by activity
        split_ep = split_dataframe(episode, encoding_target)

        if target_values[0] > 0:
            split_ep = split_ep[1:]

        n_intra_eps = len(split_ep)

        intra_episodes = dict()
        partial_ep_counter = 0
        for i in np.arange(n_intra_eps):
            # get only neg activity
            if i % 2 == 1:
                continue

            if i == n_intra_eps - 1:
                partial_episode = split_ep[i]
                partial_episode.reset_index(drop=True, inplace=True)
            else:
                pt_neg_activity = split_ep[i]
                pt_pos_activity = split_ep[i + 1].head(max_event_duration)

                partial_episode = pd.concat([pt_neg_activity, pt_pos_activity])

                if partial_episode.shape[0] < min_ep_duration:
                    continue

                partial_episode.reset_index(drop=True, inplace=True)

            intra_episodes[partial_ep_counter] = partial_episode
            partial_ep_counter += 1

        return intra_episodes

    @staticmethod
    def check_any_event(episode: pd.DataFrame,
                        target_variable: str):
        """
        Checking whether the episode contains positive activity
        :param episode:
        :param target_variable:
        :return:
        """
        final_target = episode[target_variable].values[-1]

        is_positive = final_target > 0

        return is_positive

    @staticmethod
    def non_overlapping_resample(episode: pd.DataFrame,
                                 target_variable: str,
                                 sample_interval_size: int,
                                 include_class_condition: bool = False):
        """
        Subsetting an episode every sample_interval_size instances

        :param episode: episode as pd.df
        :param target_variable: (str) target var
        :param sample_interval_size: (int) sampling frequency
        :param include_class_condition: (bool) whether or not to sample in the minority class
        :return:
        """
        target = episode[target_variable].values
        ind = np.arange(len(target))

        time_condition = ind % sample_interval_size == 0

        if include_class_condition:
            class_condition = target == 1
        else:
            class_condition = time_condition.copy()

        sampled_observations = time_condition | class_condition

        sampled_episode = episode.iloc[sampled_observations, :]
        sampled_episode = sampled_episode.reset_index(drop=True)

        return sampled_episode

    def subset_episodes(self,
                        dataset: Dict,
                        ind: List,
                        resample_episodes: bool = False,
                        resample_on_positives: bool = True,
                        resample_size: Optional[int] = None):
        """
        Subset data set by indices, and split episodes

        :param dataset: dictionary of pd.df episodes
        :param ind: indices of episodes to subset
        :param resample_episodes: (bool) whether or not to resample episodes - remove overlapping observations
        :param resample_on_positives: (bool) whether or not to resample on positive values
        :param resample_size: (int) resample periodicity

        :return: dictionary with subset of episodes
        """

        entity_ids = [*dataset]

        sub_entity_ind = [entity_ids[x] for x in ind]
        data_subset = dict_subset_by_key(dataset, sub_entity_ind)
        data_subset_sp = self.split_episodes_in_db(data_subset)

        if resample_episodes:
            for k in data_subset_sp:
                data_subset_sp[k] = \
                    self.non_overlapping_resample(episode=data_subset_sp[k],
                                                  target_variable=self.target_variable,
                                                  sample_interval_size=resample_size,
                                                  include_class_condition=resample_on_positives)

        return data_subset_sp
