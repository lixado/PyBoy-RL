from pyboy.pyboy import *
from AISettings.AISettingsInterface import AISettingsInterface
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

        done = pyboy_done or self.pyboy.game_wrapper().game_over()
        return observation, reward, done, info

    def setAISettings(self, aisettings: AISettingsInterface):
        self.aiSettings = aisettings

    def reset(self):
        """ Reset (or start) the gym environment throught the game_wrapper """
        if not self._started:
            self.game_wrapper.start_game(**self._kwargs)
            self._started = True
        else:
            self.game_wrapper.reset_game()

        # release buttons if not pressed now but were pressed in the past
        for pressedFromBefore in [pressed for pressed in self._button_is_pressed if self._button_is_pressed[pressed] == True]: # get all buttons currently pressed
            self.pyboy.send_input(self._release_button[pressedFromBefore])
        self.button_is_pressed = {button: False for button in self._buttons} # reset all buttons

        return self._get_observation()