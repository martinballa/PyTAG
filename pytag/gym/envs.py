import gymnasium as gym
import numpy as np
import jpype

from pytag.pyTAG import PyTAG, MultiAgentPyTAG
from abc import abstractmethod
from typing import Dict, List, Union

class TagSingleplayerGym(gym.Env):
    def __init__(self, game_id: str, agent_ids: List[str], seed: int=0, obs_type:str="vector"):
        super().__init__()
        self._last_obs_vector = None
        self._last_action_mask = None
        self._obstype = obs_type
        
        # Initialize the java environment
        self._env = PyTAG(agent_ids=agent_ids, game_id=game_id, seed=seed, isNormalized=True)
        assert agent_ids.count("python") == 1, "Only one python agent is allowed - look at TAGMultiplayerGym for multiplayer support"
        self._playerID = agent_ids.index("python")

        # Construct action/observation space
        self._env.reset()
        self.action_space = gym.spaces.Discrete(self._env.action_space)

        obs_size = int(self._env.observation_space)
        self.observation_space = gym.spaces.Box(shape=(obs_size,), low=float("-inf"), high=float("inf"))
        self._action_tree_shape = self._env.get_action_tree_shape()

    def get_action_tree_shape(self):
        return self._action_tree_shape
    
    def reset(self):
        self._env.reset()
        self._update_data()
        
        return self._last_obs_vector, {"action_tree": self._action_tree_shape, "action_mask": self._last_action_mask, "has_won": int(str(self._java_env.getPlayerResults()[self._playerID]) == "WIN_GAME")}
    
    def step(self, action):
        # Verify
        if not self.is_valid_action(action):
            # Execute a random action
            valid_actions = np.where(self._last_action_mask)[0]
            action = self.np_random.choice(valid_actions)
            self._env.step(action)
            reward = -1
        else:
            self._env.step(action)
            reward = int(str(self._java_env.getPlayerResults()[self._playerID]) == "WIN_GAME")
            if str(self._java_env.getPlayerResults()[self._playerID]) == "LOSE_GAME": reward = -1

        self._update_data()
        done = self._java_env.isDone()
        truncated = False
        info = {"action_mask": self._last_action_mask,
                "has_won": int(str(self._java_env.getPlayerResults()[self._playerID]) == "WIN_GAME")}
        return self._last_obs_vector, reward, done, truncated, info
    
    def close(self):
        pass
    
    def is_valid_action(self, action: int) -> bool:
        return self._last_action_mask[action] 
    
    def _update_data(self):
        """Updates the observation and action mask.
        """

        if self._obstype == "vector":
            obs = self._java_env.getObservationVector()
            self._last_obs_vector = np.array(obs, dtype=np.float32)
        elif self._obstype == "json":
            obs = self._java_env.getObservationJson()
            self._last_obs_vector = obs
        
        action_mask = self._java_env.getActionMask()
        self._last_action_mask = np.array(action_mask, dtype=bool)
    
"""Multi-agent environment for TAG."""
class TAGMultiplayerGym(gym.Env):
    def __init__(self, game_id: str, agent_ids: List[str], seed: int=0, obs_type:str="vector"):
        super().__init__()
        self._java_env = MultiAgentPyTAG(game_id=game_id, agent_ids=agent_ids, seed=0, obs_type=obs_type)

    def reset(self):
        # return {"player1": obs1, "player2": obs2}
        obs, info = self._java_env.reset()
        return obs, info

        
    def step(self, actions: Dict[str, int]):
        obs, reward, done, truncated, info = self._java_env.step(actions)
        return obs, reward, done, truncated, info
        # return {"player1": obs1, "player2": obs2}
        
    def close(self):
        pass