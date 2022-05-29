from typing import Any, List
from numpy import float64
from pyboy import PyBoy, WindowEvent

class GameState():
    def __init__(self, pyboy: PyBoy):
        """Used to hold a copy of the previous game state"""
        raise Exception("GameState init not implemented!")

class Config():
    def __init__(self):
        self.exploration_rate = 1 # explore 100% of times, this value is changed in the act function
        self.exploration_rate_decay = 0.9999995 #0.99999975
        self.exploration_rate_min = 0.001

        """
            Memory
        """
        self.deque_size = 400000
        self.batch_size = 256
        self.save_every = 5e5  # no. of experiences between saving Mario Net

        """
            Q learning
        """
        self.gamma = 0.9
        self.learning_rate = 0.000250
        self.learning_rate_decay = 0.99999985
        self.burnin = 1e4  # min. experiences before training
        self.learn_every = 3  # no. of experiences between updates to Q_online
        self.sync_every = 1e3  # no. of experiences between Q_target & Q_online sync


class AISettingsInterface:
    def GetReward(self, prevGameState: Any, pyboy: PyBoy) -> float64:
        """Reward function for the AI"""
        raise Exception("GetReward not implemented!")

    def GetActions(self) -> List[WindowEvent]:
        """Get action space for AI"""
        raise Exception("GetActions not implemented!")

    def GetGameState(self, pyboy: PyBoy) -> GameState:
        """Get game state from pyboy to save important information"""
        raise Exception("GetGameState not implemented!")

    def GetLength(self, pyboy: PyBoy) -> Any:
        """Used to plot x position of player, returns 0 as default"""
        raise 0

    def GetHyperParameters(self) -> Config:
        """Used to get hyperparameters"""
        return Config() # return default

    def GetBossHyperParameters(self) -> Config:
        """Used to get boss model hyperparameters"""
        return Config() # return default

    def PrintGameState(self, pyboy: PyBoy) -> None:
        """Used to print in playtest mode"""
        raise Exception("PrintGameState not implemented!")

    def IsBossActive(self, pyboy: PyBoy) -> bool:
        """Returns true if boss is active, else false"""
        return False # return false as Default 
