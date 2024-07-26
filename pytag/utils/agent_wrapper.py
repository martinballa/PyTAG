import random
import time

import jpype
from jpype import JImplements, JOverride

pythonActor = jpype.JClass("players.python.Actor")
pythonAgent = jpype.JClass("players.python.PythonAgent")

class RandomAgent():
    def __init__(self):
        pass

    def getAction(self, gameState, possibleActions):
        rnd_action = random.choice(possibleActions)
        return rnd_action

def get_wrapped_agent(agent, delay_ms=-1):
    """
    agent - object which is used for action selection
    delay_ms - used to slow done the decision making - useful for visualisation
    """
    # agent is the agent object which is
    wrapped_actor = Actor(agent, delay_ms)
    return pythonAgent(wrapped_actor)

@JImplements(pythonActor)
class Actor:
    """
    Interface used to implement the Actor interface that can be used for action selection in TAG
    """
    def __init__(self, agent, delay_ms=-1):
        self.agent = agent
        self.delay_ms = delay_ms

    @JOverride
    def getAction(self, gameState, possibleActions):
        if self.delay_ms > 0:
            time.sleep(self.delay_ms)
        action = self.agent.getAction(gameState, possibleActions)
        return action
