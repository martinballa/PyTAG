import time
import json
from collections import deque
from typing import Optional

import numpy as np
import torch


import gymnasium as gym
from gymnasium.vector import VectorEnvWrapper


class MergeActionMaskWrapper(VectorEnvWrapper):
    def reset_wait(self, **kwargs):
        obs, infos = self.env.reset_wait(**kwargs)
        return obs, self._merge_action_masks(infos)
    
    def step_wait(self):
        obs, rewards, dones, truncated, infos = self.env.step_wait()
        return obs, rewards, dones, truncated, self._merge_action_masks(infos)
    
    def _merge_action_masks(self, infos):
        infos["action_mask"] = np.stack(infos["action_mask"])
        del infos["_action_mask"] # Not needed
        return infos

class StrategoWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(27, 10, 10), dtype=np.float32)
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info
    def observation(self, observation):
        observation_ = torch.from_numpy(observation.reshape(10, 10)).to(torch.int64)
        observation_ = torch.nn.functional.one_hot(observation_+13, num_classes=27)
        observation_ = observation_.permute(2, 0, 1).float()
        return observation_

class SushiGoWrapper(gym.ObservationWrapper):
    # Sushi GO wrapper - an example that extracts the observation from JSON
    def __init__(self, env, n_players=2):
        super().__init__(env)
        self.card_types = ["Maki", "Maki-2", "Maki-3", "Chopsticks", "Tempura", "Sashimi", "Dumpling", "SquidNigiri",
                      "SalmonNigiri", "EggNigiri", "Wasabi", "Pudding"]
        self.n_players = n_players
        self._obs_shapes = [-1, 147, 160, 173]
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=[self._obs_shapes[n_players-1]], dtype=np.float32)
        self.max_cards_in_hand = 10
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info
    def observation(self, observation):
        obs = self.process_json_obs(observation)
        obs = torch.from_numpy(obs).float()
        return obs

    def get_card_id(self, card):
        card_emb = np.zeros(len(self.card_types))
        if card != "EmptyDeck":
            card_emb[self.card_types.index(card)] = 1
        return card_emb

    def process_json_obs(self, json_obs, normalise=True):
        # actions represent cardIds from left to right
        json_ = json.loads(str(json_obs))
        player_id = json_["PlayerID"]
        played_cards = json_["playedCards"].split(",")
        cards_in_hand = json_["cardsInHand"].split(",")
        score = json_["playerScore"] / 50  # keep it close to 0-1
        round = json_["rounds"] / 3  # max 3 rounds

        opp_scores = []
        opponent_played_cards_ = []
        for key in json_.keys():
            if f"opp" in key and "playedCards" in key:
                opp_played_cards = json_[key].split(",")
                opponent_played_cards_.append(([self.get_card_id(card) for card in opp_played_cards]))
            if f"opp" in key and "score" in key:
                opp_score = json_[key] / 50
                opp_scores.append(opp_score)

        played_cards_ = [self.get_card_id(card) for card in played_cards]
        cards_in_hand_ = [self.get_card_id(card) for card in cards_in_hand]
        while len(cards_in_hand_) < self.max_cards_in_hand:
            cards_in_hand_.append(np.zeros(len(self.card_types)))

        score = np.expand_dims(score, 0)
        round = np.expand_dims(round, 0)
        played_cards = np.sum(played_cards_, axis=0)
        cards_in_hand = np.stack(cards_in_hand_, 0).flatten()
        opp_played_cards = np.concatenate([np.sum(opc, axis=0) for opc in opponent_played_cards_])
        obs = np.concatenate([score, round, played_cards, cards_in_hand, opp_played_cards, opp_scores])
        return obs

class RecordEpisodeStatistics(gym.Wrapper):
    # Based on RecordEpisodeStatistics from gymnasium, but it checks whether the player has won the game
    """This wrapper will keep track of cumulative rewards and episode lengths.

    At the end of an episode, the statistics of the episode will be added to ``info``
    using the key ``episode``. If using a vectorized environment also the key
    ``_episode`` is used which indicates whether the env at the respective index has
    the episode statistics.
    """

    def __init__(self, env: gym.Env, deque_size: int = 100):
        """This wrapper will keep track of cumulative rewards and episode lengths.

        Args:
            env (Env): The environment to apply the wrapper
            deque_size: The size of the buffers :attr:`return_queue` and :attr:`length_queue`
        """
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_count = 0
        self.episode_start_times: np.ndarray = None
        self.episode_returns: Optional[np.ndarray] = None
        self.episode_lengths: Optional[np.ndarray] = None
        self.episode_wins: Optional[np.ndarray] = None
        self.return_queue = deque(maxlen=deque_size)
        self.length_queue = deque(maxlen=deque_size)
        self.win_queue = deque(maxlen=deque_size)
        self.is_vector_env = getattr(env, "is_vector_env", False)

    def reset(self, **kwargs):
        """Resets the environment using kwargs and resets the episode returns and lengths."""
        obs, info = super().reset(**kwargs)
        self.episode_start_times = np.full(
            self.num_envs, time.perf_counter(), dtype=np.float32
        )
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.episode_wins = np.zeros(self.num_envs, dtype=np.int32)
        return obs, info

    def step(self, action):
        """Steps through the environment, recording the episode statistics."""
        (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        ) = self.env.step(action)
        assert isinstance(
            infos, dict
        ), f"`info` dtype is {type(infos)} while supported dtype is `dict`. This may be due to usage of other wrappers in the wrong order."
        self.episode_returns += rewards
        self.episode_lengths += 1
        self.episode_wins += infos["has_won"]
        dones = np.logical_or(terminations, truncations)
        num_dones = np.sum(dones)
        if num_dones:
            if "episode" in infos or "_episode" in infos:
                raise ValueError(
                    "Attempted to add episode stats when they already exist"
                )
            else:
                outcomes = np.array([0 if final_inf is None else final_inf["has_won"] for final_inf in infos["final_info"]])
                infos["episode"] = {
                    "r": np.where(dones, self.episode_returns, 0.0),
                    "l": np.where(dones, self.episode_lengths, 0),
                    "w": outcomes,
                    "t": np.where(
                        dones,
                        np.round(time.perf_counter() - self.episode_start_times, 6),
                        0.0,
                    ),
                    "wins": np.where(outcomes == 1.0, 1.0, 0.0),
                    "losses": np.where(outcomes == -1.0, 1.0, 0.0),
                    "ties": np.where(outcomes == 0, 1.0, 0.0),
                }
                if self.is_vector_env:
                    infos["_episode"] = np.where(dones, True, False)
            self.return_queue.extend(self.episode_returns[dones])
            self.length_queue.extend(self.episode_lengths[dones])
            self.episode_count += num_dones
            self.episode_lengths[dones] = 0
            self.episode_returns[dones] = 0
            self.episode_start_times[dones] = time.perf_counter()
        return (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        )

# Adapted wrapper for the self-play case
class RecordSelfPlayEpStats(gym.Wrapper):
    # Based on RecordEpisodeStatistics from gymnasium, but it checks whether the player has won the game
    """This wrapper will keep track of cumulative rewards and episode lengths.

    At the end of an episode, the statistics of the episode will be added to ``info``
    using the key ``episode``. If using a vectorized environment also the key
    ``_episode`` is used which indicates whether the env at the respective index has
    the episode statistics.
    """

    def __init__(self, env: gym.Env):
        """This wrapper will keep track of cumulative rewards and episode lengths.

        Args:
            env (Env): The environment to apply the wrapper
            deque_size: The size of the buffers :attr:`return_queue` and :attr:`length_queue`
        """
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_count = 0
        self.episode_start_times: np.ndarray = None
        self.episode_returns: Optional[np.ndarray] = None
        self.episode_lengths: Optional[np.ndarray] = None
        self.episode_wins: Optional[np.ndarray] = None
        self.total_lenghts: Optional[np.ndarray] = None
        self.is_vector_env = getattr(env, "is_vector_env", False)

    def reset(self, **kwargs):
        """Resets the environment using kwargs and resets the episode returns and lengths."""
        obs, info = super().reset(**kwargs)
        self.episode_start_times = np.full(
            self.num_envs, time.perf_counter(), dtype=np.float32
        )
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.episode_wins = np.zeros(self.num_envs, dtype=np.int32)
        self.total_lenghts = np.zeros(self.num_envs, dtype=np.int32)
        return obs, info

    def step(self, action):
        """Steps through the environment, recording the episode statistics."""
        (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        ) = self.env.step(action)
        assert isinstance(
            infos, dict
        ), f"`info` dtype is {type(infos)} while supported dtype is `dict`. This may be due to usage of other wrappers in the wrong order."
        update_idx = infos["player_id"] == infos["learning_player"]
        self.episode_returns += rewards * update_idx
        self.episode_lengths += update_idx
        self.total_lenghts += 1
        self.episode_wins += infos["has_won"] * update_idx
        dones = np.logical_or(terminations, truncations)
        num_dones = np.sum(dones)
        if num_dones:
            if "episode" in infos or "_episode" in infos:
                raise ValueError(
                    "Attempted to add episode stats when they already exist"
                )
            else:
                outcomes = np.array([0 if final_inf is None else final_inf["has_won"] for final_inf in infos["final_info"]])
                infos["episode"] = {
                    "r": np.where(dones, self.episode_returns, 0.0),
                    "l": np.where(dones, self.episode_lengths, 0),
                    "total_l": np.where(dones, self.total_lenghts, 0),
                    "w": outcomes,
                    "t": np.where(
                        dones,
                        np.round(time.perf_counter() - self.episode_start_times, 6),
                        0.0,
                    ),
                    "wins": np.where(outcomes == 1.0, 1.0, 0.0),
                    "losses": np.where(outcomes == -1.0, 1.0, 0.0),
                    "ties": np.where(outcomes == 0, 1.0, 0.0),
                }
                if self.is_vector_env:
                    infos["_episode"] = np.where(dones, True, False)
            self.episode_count += num_dones
            self.episode_lengths[dones] = 0
            self.episode_returns[dones] = 0
            self.total_lenghts[dones] = 0
            self.episode_start_times[dones] = time.perf_counter()
        return (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        )