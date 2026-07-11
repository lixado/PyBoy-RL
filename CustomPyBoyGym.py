from pyboy.pyboy import *
from AISettings.AISettingsInterface import AISettingsInterface
import gym

# Adopted from https://github.com/NicoleFaye/PyBoy/blob/rl-test/PokemonPinballEnv.py
from gym import spaces
import numpy as np
from pyboy import PyBoy

actions = ['','a', 'b', 'left', 'right', 'up', 'down', 'start', 'select']

matrix_shape = (16, 20)
game_area_observation_space = spaces.Box(low=0, high=255, shape=matrix_shape, dtype=np.uint8)

class PyBoyGymEnv(gym.Env):

    def __init__(self, pyboy, debug=False):
        super().__init__()
        self.pyboy = pyboy
        self._fitness=0
        self._previous_fitness=0
        self.debug = debug
        self._started = False

        if not self.debug:
            self.pyboy.set_emulation_speed(0)

        self.action_space = spaces.Discrete(len(actions))
        self.observation_space = game_area_observation_space
        self.pyboy.game_wrapper.start_game()

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        # Move the agent
        if action == 0:
            pass
        else:
            self.pyboy.button(actions[action])

        # Consider disabling renderer when not needed to improve speed:
        # self.pyboy.tick(1, False)
        self.pyboy.tick(1)

        done = self.pyboy.game_wrapper.game_over

        self._calculate_fitness()
        reward=self._fitness-self._previous_fitness

        observation=self.pyboy.game_area()
        info = {}
        truncated = False

        return observation, reward, done, truncated, info

    def _calculate_fitness(self):
        self._previous_fitness=self._fitness

        # NOTE: Only some game wrappers will provide a score
        # If not, you'll have to investigate how to score the game yourself
        self._fitness=self.pyboy.game_wrapper.score

    def reset(self):
        self.pyboy.game_wrapper.reset_game()
        self._fitness=0
        self._previous_fitness=0

        observation=self.pyboy.game_area()
        info = {}
        return observation, info

    def render(self, mode='human'):
        pass

    def close(self):
        self.pyboy.stop()

class CustomPyBoyGym(PyBoyGymEnv):
    def step(self, list_actions):
        """
            Simultanious action implemention
        """
        info = {}

        previousGameState = self.aiSettings.GetGameState(self.pyboy)

        if list_actions[0] == self._DO_NOTHING:
            pyboy_done = self.pyboy.tick()
        else:
            # release buttons if not pressed now but were pressed in the past
            for pressedFromBefore in [pressed for pressed in self._button_is_pressed if self._button_is_pressed[pressed] == True]: # get all buttons currently pressed
                if pressedFromBefore not in list_actions:
                    release = self._release_button[pressedFromBefore]
                    self.pyboy.send_input(release)
                    self._button_is_pressed[release] = False

            # press buttons we want to press
            for buttonToPress in list_actions:
                self.pyboy.send_input(buttonToPress)
                self._button_is_pressed[buttonToPress] = True # update status of the button

            pyboy_done = self.pyboy.tick()

        # reward 
        reward = self.aiSettings.GetReward(previousGameState, self.pyboy)

        observation = self._get_observation()

        done = pyboy_done or self.pyboy.game_wrapper.game_over()
        return observation, reward, done, info

    def setAISettings(self, aisettings: AISettingsInterface):
        self.aiSettings = aisettings

    def reset(self):
        """ Reset (or start) the gym environment throught the game_wrapper """
        if not self._started:
            self.pyboy.game_wrapper.start_game()
            self._started = True
        else:
            self.pyboy.game_wrapper.reset_game()

        # release buttons if not pressed now but were pressed in the past
        for pressedFromBefore in [pressed for pressed in self._button_is_pressed if self._button_is_pressed[pressed] == True]: # get all buttons currently pressed
            self.pyboy.send_input(self._release_button[pressedFromBefore])
        self.button_is_pressed = {button: False for button in self._buttons} # reset all buttons

        return self._get_observation()