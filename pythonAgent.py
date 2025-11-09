# wraps a python agent into a java class and executes it in game
import jpype
import random

from pytag.pyTAG import PyTAG, MultiAgentPyTAG, list_supported_games
import os
# from pytag.players import PythonAgent
from pytag.utils.agent_wrapper import RandomAgent, get_wrapped_agent

# PythonAgent = jpype.JClass("players.python.PythonAgent")
# Game = jpype.JClass("core.Game")

# this is the correct way to specify a game
# todo write an entry point where we can supply our custom agent from Python
# Game.main(["game=LoveLetter"])


agent_ids = ["python", "random", "random"]
env = PyTAG(agent_ids, "Catan", obs_type="json")

# todo we could hide this part and just have the user submit their player
wrapped_agent = get_wrapped_agent(RandomAgent(), -1)

env.evaluate(wrapped_agent, useGUI=True, repetitions=10)


