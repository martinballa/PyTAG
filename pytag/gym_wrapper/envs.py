# Gym wrappers for PyTAG
import gymnasium as gym

from pytag import PyTAG, MultiAgentPyTAG
from typing import Dict, List

class TagSingleplayerGym(gym.Env):
    def __init__(self, game_id: str, agent_ids: List[str], seed: int=0, obs_type:str="vector"):
        super().__init__()
        self._obs_type = obs_type
        
        # initialise a single player pytag environment
        self._env = PyTAG(agent_ids=agent_ids, game_id=game_id, seed=seed, obs_type=obs_type)
        assert agent_ids.count("python") == 1, "Only one python agent is allowed - look at TAGMultiplayerGym for multiplayer support"
        self._playerID = agent_ids.index("python")

        # Construct action/observation space - note this requires resetting the environment
        self._env.reset()
        self.action_space = gym.spaces.Discrete(self._env.action_space)

        obs_size = int(self._env.observation_space)
        self.observation_space = gym.spaces.Box(shape=(obs_size,), low=float("-inf"), high=float("inf"))
        self._action_tree_shape = self._env.get_action_tree_shape()

    def get_action_tree_shape(self):
        return self._action_tree_shape

    def sample_rnd_action(self):
        return self._env.sample_rnd_action()
    
    def reset(self):
        obs, info = self._env.reset()
        return obs, info

    def step(self, action):
        obs, reward, done, info =  self._env.step(action)
        # truncated = False
        return obs, reward, done, False, info
    
    def close(self):
        pass
    
    def is_valid_action(self, action: int) -> bool:
        return self._last_action_mask[action]
    
"""Multi-agent environment for TAG."""
class TAGMultiplayerGym(gym.Env):
    def __init__(self, game_id: str, agent_ids: List[str], seed: int=0, obs_type:str="vector"):
        super().__init__(game_id, agent_ids, seed, obs_type)
        self._java_env = MultiAgentPyTAG(game_id=game_id, agent_ids=agent_ids, seed=0, obs_type=obs_type)

    def reset(self):
        # return {"player1": obs1, "player2": obs2}
        obs, info = self._java_env.reset()
        return obs, info

        
    def step(self, actions: Dict[str, int]):
        # return {"player1": obs1, "player2": obs2}
        obs, reward, done, truncated, info = self._java_env.step(actions)
        return obs, reward, done, truncated, info
        
    def close(self):
        pass