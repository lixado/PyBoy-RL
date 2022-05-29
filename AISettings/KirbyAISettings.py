import itertools
from pyboy import WindowEvent
from AISettings.AISettingsInterface import AISettingsInterface
from AISettings.AISettingsInterface import Config


class GameState:
    def __init__(self, pyboy):
        game_wrapper = pyboy.game_wrapper()
        self.boss_health = pyboy.get_memory_value(0xD093)
        self.screen_x_position = pyboy.get_memory_value(0xD053)
        self.kirby_x_position = pyboy.get_memory_value(0xD05C)
        self.kirby_y_position = pyboy.get_memory_value(0xD05D)
        self.game_state = pyboy.get_memory_value(0xD02C)
        scx = pyboy.botsupport_manager().screen().tilemap_position_list()[16][0]
        self.level_progress = self.screen_x_position * 16 + (scx - 7) % 16 + self.kirby_x_position
        self.health = game_wrapper.health
        self.lives_left = game_wrapper.lives_left
        self.score = game_wrapper.score


class KirbyAI(AISettingsInterface):
    def GetReward(self, previous_kirby: GameState, pyboy):
        current_kirby = GameState(pyboy)

        if current_kirby.boss_health == 0 and previous_kirby.boss_health > 0:
            return 10000

        if current_kirby.boss_health < previous_kirby.boss_health:
            return 1000

        if current_kirby.health < previous_kirby.health and current_kirby.health == 1:
            return -100

        if current_kirby.health == 0 and previous_kirby.health != 0:
            return -1000

        if current_kirby.health > 0 and current_kirby.game_state == 6 and previous_kirby.game_state != 6:  # if reached warpstar
            return 1000

        if not self.IsBossActive(pyboy) and current_kirby.game_state != 6:  # if boss is dead or not active, punish for not moving right
            if current_kirby.kirby_x_position < previous_kirby.kirby_x_position:  # moving left
                return -1

            if current_kirby.level_progress != previous_kirby.level_progress and current_kirby.kirby_x_position == 68:  # moving most left
                return -5

            if current_kirby.level_progress == previous_kirby.level_progress:  # standing still
                return -1

            if current_kirby.kirby_x_position == 76:  # moving most right
                return 5
            return 1  # moving right
        else:
            if current_kirby.score>previous_kirby.score:
                return 100
        return 0

    def GetActions(self):
        baseActions = [WindowEvent.PRESS_BUTTON_A,
                       WindowEvent.PRESS_BUTTON_B,
                       WindowEvent.PRESS_ARROW_UP,
                       WindowEvent.PRESS_ARROW_DOWN,
                       WindowEvent.PRESS_ARROW_LEFT,
                       WindowEvent.PRESS_ARROW_RIGHT
                       ]

        totalActionsWithRepeats = list(itertools.permutations(baseActions, 2))
        withoutRepeats = []

        for combination in totalActionsWithRepeats:
            reversedCombination = combination[::-1]
            if (reversedCombination not in withoutRepeats):
                withoutRepeats.append(combination)

        filteredActions = [[action] for action in baseActions] + withoutRepeats

        return filteredActions

    def PrintGameState(self, pyboy):
        pass

    def GetGameState(self, pyboy) -> GameState:
        return GameState(pyboy)

    def GetLength(self, pyboy):
        return self.GetGameState(pyboy).boss_health

    def IsBossActive(self, pyboy):
        if self.GetGameState(pyboy).boss_health > 0:
            return True
        return False

    def GetHyperParameters(self) -> Config:
        config = Config()
        config.exploration_rate_decay = 0.9999975
        config.exploration_rate_min = 0.01
        config.deque_size = 500000
        config.batch_size = 64
        config.save_every = 2e5
        config.learning_rate_decay = 0.9999985
        config.gamma = 0.8
        config.learning_rate = 0.0002
        config.burnin = 1000
        config.sync_every = 100
        return config

    def GetBossHyperParameters(self) -> Config:
        config = self.GetHyperParameters()
        config.exploration_rate_decay = 0.99999975
        return config
