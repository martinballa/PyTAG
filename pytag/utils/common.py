# various helper functions
import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation as FrameStack
import numpy as np
import torch

import jpype
import jpype.imports
from pytag.utils.wrappers import StrategoWrapper, SushiGoWrapper


def make_env(env_id, seed, opponent, n_players, framestack=1, obs_type="vector"):
    def thunk():
        # always have a python agent first (at least in our experiments)
        agent_ids = ["python"]
        for i in range(n_players - 1):
            agent_ids.append(opponent)
        # obs_type = "json" if "Sushi" in env_id else "vector" # , obs_type=obs_type
        env = gym.make(env_id, seed=seed, agent_ids=agent_ids, obs_type=obs_type)
        if "Stratego" in env_id:
            env = StrategoWrapper(env)
        if "Sushi" in env_id:
            env = SushiGoWrapper(env)
        if framestack > 1:
            env = FrameStack(env, framestack)
        return env
    return thunk

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer