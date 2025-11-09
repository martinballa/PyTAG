# Multi-agent example of running 2 python based random agents against each other
import time
from pytag.pyTAG import MultiAgentPyTAG, list_supported_games

if __name__ == "__main__":
    # multi-agent example
    EPISODES = 100
    players = ["python", "python", "python"]
    supported_games = list_supported_games()
    env = MultiAgentPyTAG(players, game_id="Catan", obs_type="json")
    done = False

    # these are to keep track of agent specific statistics
    # in this example these are not reset between episodes
    start_time = time.time()
    steps = [0] * len(players)
    wins = [0] * len(players)
    rewards = [0] * len(players)
    # main MA RL loop
    for e in range(EPISODES):
        obs, info = env.reset()
        done = False
        while not done:
            player_id = env.getPlayerID()
            steps[player_id] += 1

            rnd_action = env.sample_rnd_action()
            obs, reward, done, info = env.step(rnd_action)
            player_id = env.getPlayerID()

            # rewards are returned for both players at each step - these may be just the terminal rewards
            for p_id in range(len(players)):
                rewards[p_id] += reward[p_id]

            if done:
                # determine the winner at the final state
                for p_id in range(len(players)):
                    if env.has_won(p_id):
                        wins[p_id] += 1
                print(f"Game over rewards {rewards} in {steps} steps results =  {wins}")
                break
    env.close()