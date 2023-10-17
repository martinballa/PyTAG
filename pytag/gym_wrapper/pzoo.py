from typing import List

import gymnasium as gym
from pettingzoo import AECEnv
from pytag import MultiAgentPyTAG

class AECPyTAG(AECEnv):
    metadata = {
        "name": "custom_environment_v0",
    }

    def __init__(self, game_id: str, n_players: int = 2, seed: int=0, obs_type:str="vector"):
        agent_ids = ["python"] * n_players
        self._env = MultiAgentPyTAG(game_id=game_id, agent_ids=agent_ids, seed=0, obs_type=obs_type)
        # need to reset in order to work this out
        self._env.reset()
        self.action_space = gym.spaces.Discrete(self._env.action_space)

        obs_size = int(self._env.observation_space)
        self.observation_space = gym.spaces.Box(shape=(obs_size,), low=float("-inf"), high=float("inf"))
        self._action_tree_shape = self._env.get_action_tree_shape()

        # petting zoo
        self.possible_agents = ["player_" + str(i) for i in range(n_players)]
        self.agents = ["player_" + str(i) for i in range(n_players)]

    def reset(self, seed=None, options=None):
        return self._env.reset()

    def step(self, actions):
        return self._env.step(actions)

    def render(self):
        # not implemented
        pass

    def observation_space(self, agent):
        return self.observation_spaces #[agent]

    def action_space(self, agent):
        return self.action_spaces #[agent]

if __name__ == "__main__":
    from pettingzoo.test import parallel_api_test
    env = AECPyTAG(game_id="Diamant", n_players=2)
    parallel_api_test(env, num_cycles=100)