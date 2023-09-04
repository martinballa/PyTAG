# PyTAG usage without the gym interface
# Example that demonstrates how to use action masking manually
from pytag import PyTAG, list_supported_games
import random
import numpy as np

if __name__ == "__main__":
    # We may print all supported games - these are the ones that support the RL interface in TAG
    list_supported_games()

    list_of_agents = ["python", "random"]
    # this example shows how to sample random actions using an action mask manually
    env = PyTAG(list_of_agents, game_id="Diamant", seed=45, obs_type="vector")

    for episode in range(20):
        done = False
        obs, info = env.reset()
        while not done:
            # pick random, but valid action
            mask = np.array(info["action_mask"])
            valid_actions = np.where(mask)[0]
            action = random.choice(valid_actions)

            obs, reward, done, info = env.step(action)
            if done:
                print(f"game over player got reward {reward}")