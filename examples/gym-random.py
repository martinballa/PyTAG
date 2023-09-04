from pytag import gym_wrapper  # we need to import the gym_wrapper to register the environments as gym environments
import gymnasium as gym

env = gym.make("TAG/Diamant-v0")
obs, info = env.reset()
done = False
while not done:
    # we may invoke the built-in action sampler that uses the action mask to sample a valid action
    action = env.unwrapped.sample_rnd_action()

    obs, rewards, done, truncated, infos = env.step(action)
    print(obs, rewards, done, truncated, infos)