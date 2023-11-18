import os, random, time
import json

import jpype
import jpype.imports

import numpy as np
from typing import List
def list_supported_games(as_json=False):
    tag_jar = os.path.join(os.path.dirname(__file__), 'jars', 'ModernBoardGame.jar')
    jpype.addClassPath(tag_jar)
    if not jpype.isJVMStarted():
        jpype.startJVM(convertStrings=False)
    PyTAGEnv = jpype.JClass("core.PyTAG")
    if as_json:
        return json.loads(str(PyTAGEnv.getSupportedGamesJSON()))
    return PyTAGEnv.getSupportedGames()

def get_agent_class(agent_name):
    if agent_name == "random":
        return jpype.JClass("players.simple.RandomPlayer")
    if agent_name == "mcts":
        return jpype.JClass("players.mcts.MCTSPlayer")
    if agent_name == "osla":
        return jpype.JClass("players.simple.OSLAPlayer")
    if agent_name == "python":
        return jpype.JClass("players.python.PythonAgent")
    return None

def get_reward_types():
    return [str(rew_type) for rew_type in jpype.JClass("core.PyTAG").listRewardTypes()]

def get_mcts_with_params(json_path):
    PlayerFactory = jpype.JClass("players.PlayerFactory")
    with open(os.path.expanduser(json_path)) as json_file:
        json_string = json.load(json_file)
    json_string = str(json_string).replace('\'', '\"') # JAVA only uses " for string
    return jpype.JClass("players.mcts.MCTSPlayer")(PlayerFactory.fromJSONString(json_string))

# create the game registry when PyTAG is loaded
_game_registry = list_supported_games(as_json=True)
class PyTAG():
    """Core class to interact with the pyTAG environment. This class is a wrapper around the Java environment.
    This class expects to have a single python agents
    Note that the java jar package is expected to be in the jars folder of the same directory as this file.
    """
    def __init__(self, agent_ids: List[str], game_id: str="Diamant", seed: int=0, obs_type:str="vector", reward_type="DEFAULT"):
        self._last_obs_vector = None
        self._last_action_mask = None
        self._rnd = random.Random(seed)
        self._obs_type = obs_type
        self._reward_type = reward_type
        self._prev_score = 0.0  # kepts for calculating delta scores for reward

        assert game_id in _game_registry, f"Game {game_id} not supported. Supported games are {_game_registry}"
        assert _game_registry[game_id][obs_type] == True, f"Game {game_id} does not support observation type {obs_type}"
        assert reward_type in get_reward_types(), f"Reward type {reward_type} not supported. Supported reward types are {get_reward_types()}"
        # start up the JVM
        tag_jar = os.path.join(os.path.dirname(__file__), 'jars', 'ModernBoardGame.jar')
        jpype.addClassPath(tag_jar)
        if not jpype.isJVMStarted():
            jpype.startJVM(convertStrings=False)

        # access to the java classes
        PyTAGEnv = jpype.JClass("core.PyTAG")
        Utils = jpype.JClass("utilities.Utils")
        GameType = jpype.JClass("games.GameType")

        # Initialize the java environment
        gameType = GameType.valueOf(Utils.getArg([""], "game", game_id))

        if agent_ids[0] == "mcts":
            agents = [get_mcts_with_params(f"~/data/pyTAG/MCTS_for_{game_id}.json")() for agent_id in agent_ids]
        else:
            agents = [get_agent_class(agent_id)() for agent_id in agent_ids]
        self._playerID = agent_ids.index("python") # if multiple python agents this is the first one
        rew_type = PyTAGEnv.getRewardType(self._reward_type)
        self._java_env = PyTAGEnv(gameType, None, jpype.java.util.ArrayList(agents), seed, True, rew_type)

        # Construct action/observation space
        self._java_env.reset()
        action_mask = self._java_env.getActionMask()
        num_actions = len(action_mask)
        self.action_space = num_actions

        obs_size = int(self._java_env.getObservationSpace())
        self.observation_space = obs_size # (obs_size,)
        self._action_tree_shape = 1

    def get_action_tree_shape(self):
        return self._action_tree_shape

    def get_reward(self, player_id):
        if self._reward_type == "DEFAULT":
            return self.terminal_reward(player_id)
        # if self._reward_type == "SCORE":

        current_score = float(self._java_env.getReward(player_id))
        reward = current_score - self._prev_score
        self._prev_score = current_score
        return reward
        # return float(self._java_env.getReward(player_id))

    def reset(self):
        self._java_env.reset()
        self._prev_score = 0.0
        self._update_data()

        return self._last_obs_vector, {"action_tree": self._action_tree_shape, "action_mask": self._last_action_mask,
                                       "has_won": int(self.terminal_reward(self._playerID))}

    def step(self, action):
        # Verify
        if not self.is_valid_action(action):
            # Execute a random action
            valid_actions = np.where(self._last_action_mask)[0]
            action = self._rnd.choice(valid_actions)
            self._java_env.step(action)
            print("invalid action")
            reward = -1
        else:
            self._java_env.step(action)
            reward = self.get_reward(self._playerID)
            # reward = self.terminal_reward(self._playerID)

        self._update_data()
        done = self._java_env.isDone()
        info = {"action_mask": self._last_action_mask,
                "has_won": int(self.terminal_reward(self._playerID)),
                "final_scores": np.array(self._java_env.getScores()),
                "player_score": self._java_env.getScore(self._playerID)}
        return self._last_obs_vector, reward, done, info

    def close(self):
        jpype.shutdownJVM()

    def is_valid_action(self, action: int) -> bool:
        return self._last_action_mask[action]

    def _update_data(self):
        if self._obs_type == "vector":
            obs = self._java_env.getObservationVector()
            self._last_obs_vector = np.array(obs, dtype=np.float32)
        elif self._obs_type == "json":
            obs = self._java_env.getObservationJson()
            self._last_obs_vector = obs

        action_mask = self._java_env.getActionMask()
        self._last_action_mask = np.array(action_mask, dtype=bool)

    def get_action_mask(self):
        return self._last_action_mask

    def getVectorObs(self):
        return self._java_env.getFeatures()

    def getJSONObs(self):
        return self._java_env.getObservationJson()

    def sample_rnd_action(self):
        valid_actions = np.where(self._last_action_mask)[0]
        action = self._rnd.choice(valid_actions)
        return action


    def getPlayerID(self):
        return int(self._java_env.getPlayerID())

    def has_won(self, player_id=0):
        return int(str(self._java_env.getPlayerResults()[player_id]) == "WIN_GAME")

    def terminal_reward(self, player_id=0):
        # gets terminal reward - recommended to check if game is terminal as an unfinished game return the same value as a tie
        player_result = str(self._java_env.getPlayerResults()[player_id])
        if player_result == "WIN_GAME":
            return 1.0
        elif player_result == "LOSE_GAME":
            return -1.0
        elif player_result == "DRAW_GAME":
            return 0.5
        else:
            return 0.0

    def terminal_rewards(self):
        player_result = self._java_env.getPlayerResults()
        results = [0.0] * len(player_result)
        for i in range(len(player_result)):
            result = str(player_result[i])
            if result == "WIN_GAME":
                results[i] = 1.0
            elif result == "LOSE_GAME":
                results[i] = -1.0
            elif player_result == "DRAW_GAME":
                results[i] = 0.5
            else:
                results[i] = 0.0
        return results

class SelfPlayPyTAG(PyTAG):
    """
    Only running python agents for training - player IDs are randomised at each episode
    player ids are return in info rather than as a dictionary
    """
    def __init__(self, n_players: int, game_id: str="Diamant", seed: int=0, obs_type:str="vector", reward_type="DEFAULT"):
        super().__init__(["python"]*n_players, game_id, seed, obs_type, reward_type=reward_type)
        # the training agent always gets our internal ID representation not the ids coming from the env directly
        self._learner_id = 0
        self.n_players = n_players
        self.player_mapping = [i for i in range(n_players)]

    # def getPlayerID(self):
    #     # due to the randomisation of the player ids, we need to map the player id to the correct player
    #     playerID = super().getPlayerID()
    #     return self.player_mapping[playerID]

    def getLearnerID(self):
        return self._playerID
        # return self._learner_id

    def reset(self):
        obs, info = super().reset()
        self._playerID = self._rnd.randint(0, self.n_players-1)
        # self._rnd.shuffle(self.player_mapping)
        # self._learner_id = self._rnd.choice(self.player_mapping)
        # self._playerID = self.player_mapping[self._learner_id]
        info["player_id"] = self.getPlayerID()
        info["learning_player"] = self.getLearnerID()
        return obs, info

    def step(self, action):
        # print(f"player_id before update {self.getPlayerID()}")
        # print(f"player id = {self.getPlayerID()} and learning player = {self.getLearnerID()}")
        # print(f"action = {action}")
        # print(f"obs = \n{self._last_obs_vector.reshape(3, 3)}")

        # todo multiple calls to get_rewards leads to unexpected behaviour
        obs, rewards, done, info = super().step(action)
        info["player_id"] = self.getPlayerID()
        info["learning_player"] = self.getLearnerID()
        # info["has_won"] = int(self.terminal_reward(self.player_mapping[self._learner_id]))

        # only learner player gets rewards
        # rewards = self.get_reward(self._playerID)
        # rewards = self.terminal_reward(self._playerID)
        # rewards = self.terminal_reward(self.player_mapping[self._learner_id])

        # print("------------- Transitioning to next -------------")
        # print(f"player id = {self.getPlayerID()} and learning player = {self.getLearnerID()}")
        # print(f"last action = {action}")
        # print(f"obs = \n{obs.reshape(3, 3)}")
        # print(f"rewards = {rewards} \n\n")

        return obs, rewards, done, info




class MultiAgentPyTAG(PyTAG):
    """If there are more than one python agents, the observations are handled as dictionaries with the agent id as key.
    """
    def __init__(self, agent_ids: List[str], game_id: str="Diamant", seed: int=0, obs_type:str="vector"):
        super().__init__(agent_ids, game_id, seed, obs_type)
        self._playerIDs = []
        # collect all the player ids that are python agents
        for i, agent_id in enumerate(agent_ids):
            if agent_id == "python": self._playerIDs.append(i)
        self._last_obs_vector = {player_id: None for player_id in self._playerIDs}
        self._last_action_mask = {player_id: None for player_id in self._playerIDs}

    def _update_data(self):
        if self._obs_type == "vector":
            obs = self._java_env.getObservationVector()
            self._last_obs_vector = np.array(obs, dtype=np.float32)
        elif self._obs_type == "json":
            obs = self._java_env.getObservationJson()
            self._last_obs_vector = obs

        self._playerID = self.getPlayerID()
        action_mask = self._java_env.getActionMask()
        self._last_action_mask = np.array(action_mask, dtype=bool)

    def reset(self):
        """Resets the environment and return the initial observations for the first python agent that needs to act."""
        self._java_env.reset()
        self._playerID = self.getPlayerID()
        self._update_data()

        return {self._playerID :self._last_obs_vector}, {self._playerID:{"action_tree": self._action_tree_shape, "action_mask": self._last_action_mask,
                                       "has_won": int(self.terminal_reward(self._playerID))}}

    def step(self, action):
        """Executes the action for the current player and returns the observations for the next python agent that needs to act.
        Returns: obs, reward, done, info - obs is a dictionary with the agent id as key and the observation as value.
        reward returns the reward for all agents at the same time, done also covers all agents, info is also just for the acting player"""
        # Verify
        if not self.is_valid_action(action):
            # Execute a random action
            valid_actions = np.where(self._last_action_mask)[0]
            action = self._rnd.choice(valid_actions)
            self._java_env.step(action)
        else:
            self._java_env.step(action)

        self._update_data()
        # done is for all players
        done = self._java_env.isDone()
        info = {self._playerID: {"action_mask": self._last_action_mask,
                "has_won": int(self.terminal_reward(self._playerID))}}
        # rewards contains all the rewards for all players
        rewards = {p_id: reward for (p_id, reward) in enumerate(self.terminal_rewards())}
        return {self._playerID: self._last_obs_vector}, rewards, done, info


if __name__ == "__main__":
    # multi-agent example
    EPISODES = 100
    players = ["python", "python"]
    supported_games = list_supported_games()
    env = MultiAgentPyTAG(players, game_id="SushiGo", obs_type="json")
    done = False

    start_time = time.time()
    steps = [0] * len(players)
    wins = [0] * len(players)
    rewards = [0] * len(players)
    for e in range(EPISODES):
        obs, info = env.reset()
        done = False
        while not done:
            player_id = env.getPlayerID()
            steps[player_id] += 1

            rnd_action = env.sample_rnd_action()
            obs, reward, done, info = env.step(rnd_action)
            player_id = env.getPlayerID()
            # rewards are returned for both players
            for p_id in range(len(players)):
                rewards[p_id] += reward[p_id] # reward might go to the next player

            if done:
                for p_id in range(len(players)):
                    if env.has_won(p_id):
                        wins[p_id] += 1
                print(f"Game over rewards {rewards} in {steps} steps results =  {wins}")
                break
    env.close()

