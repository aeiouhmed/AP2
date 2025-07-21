import gym
import cv2
import numpy as np
from gym.spaces import Box

class GrayScaleObservation(gym.ObservationWrapper):
    """
    Convert the observation to grayscale.
    """
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, obs):
        """
        Convert the observation to grayscale.
        """
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        return obs

class ResizeObservation(gym.ObservationWrapper):
    """
    Resize the observation to a given shape.
    """
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)
        
        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, obs):
        """
        Resize the observation to a given shape.
        """
        obs = cv2.resize(obs, self.shape, interpolation=cv2.INTER_AREA)
        return obs

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

class FrameStack(gym.Wrapper):
    """
    Stack a number of frames together.
    """
    def __init__(self, env, num_stack):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = np.zeros((num_stack,) + env.observation_space.shape, dtype=env.observation_space.dtype)
        self.observation_space = Box(
            low=0,
            high=255,
            shape=(num_stack,) + env.observation_space.shape,
            dtype=env.observation_space.dtype
        )

    def reset(self):
        """
        Reset the environment and stack the first frame.
        """
        obs = self.env.reset()
        for _ in range(self.num_stack):
            self.frames = np.roll(self.frames, -1, axis=0)
            self.frames[-1] = obs
        return self.frames

    def step(self, action):
        """
        Step the environment and stack the new frame.
        """
        obs, reward, done, info = self.env.step(action)
        self.frames = np.roll(self.frames, -1, axis=0)
        self.frames[-1] = obs
        return self.frames, reward, done, info

class ProlongedJumpWrapper(gym.Wrapper):
    """
    Wrapper to make every jump a full-height jump by repeating the action.
    """
    def __init__(self, env, jump_duration=20):
        super().__init__(env)
        self.jump_duration = jump_duration
        self.jump_actions = {2, 4, 5, 7, 9}  # Action indices that involve jumping
        self.jump_counter = 0
        self.action_to_repeat = 0

    def step(self, action):
        # If a new jump action is initiated, start the counter
        if action in self.jump_actions and self.jump_counter == 0:
            self.jump_counter = self.jump_duration
            self.action_to_repeat = action

        # If a jump is in progress, repeat the jump action
        if self.jump_counter > 0:
            action_to_take = self.action_to_repeat
            self.jump_counter -= 1
        else:
            action_to_take = action

        obs, reward, done, info = self.env.step(action_to_take)

        # If the episode ends, reset the jump counter
        if done:
            self.jump_counter = 0
            
        return obs, reward, done, info

    def reset(self, **kwargs):
        self.jump_counter = 0
        return self.env.reset(**kwargs)
