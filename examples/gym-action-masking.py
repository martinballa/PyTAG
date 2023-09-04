# Example that demonstrates how to use action masking manually
import pytag.gym_wrapper

import random
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
import torch
import numpy as np

if __name__ == "__main__":
    # this example shows how to sample random actions using an action mask manually
    env = SyncVectorEnv([
        lambda: gym.make("TAG/Stratego", obs_type="vector")
        for i in range(1)
    ])
    
    obs, infos = env.reset()
    dones = [False]
    for i in range(200):
        # pick random, but valid action
        mask = torch.tensor(infos["action_mask"][0])
        valid_actions = np.where(mask)[0]
        action = random.choice(valid_actions)

        obs, rewards, dones, truncated, infos = env.step([action])
        if dones[0]:
            print(f"game over player got reward {rewards}")