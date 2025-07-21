import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from ..environment.wrappers import GrayScaleObservation, ResizeObservation, FrameStack, SkipFrame, ProlongedJumpWrapper
from ..environment.reward_shaping import RewardShaper

class Visualizer:
    """
    A class for visualizing the agent's gameplay.
    """
    def __init__(self, agent, filename='play.gif'):
        self.env = gym.make('SuperMarioBros-v0')
        self.env = JoypadSpace(self.env, COMPLEX_MOVEMENT)
        self.env = ProlongedJumpWrapper(self.env)
        self.env = SkipFrame(self.env, skip=4)
        self.env = GrayScaleObservation(self.env)
        self.env = ResizeObservation(self.env, shape=84)
        self.env = FrameStack(self.env, num_stack=4)
        self.env = RewardShaper(self.env)
        self.agent = agent
        self.filename = filename
        try:
            self.font = ImageFont.truetype("arial.ttf", 15)
        except IOError:
            self.font = ImageFont.load_default()

    def record(self, episode):
        """
        Record a video of the agent playing one episode.
        """
        frames = []
        state = self.env.reset()
        done = False
        total_reward = 0
        info = {'status': 'small'}
        while not done:
            frame = self.env.render(mode='rgb_array')
            img = Image.fromarray(frame)
            draw = ImageDraw.Draw(img)
            draw.text((10, 36), f"Episode: {episode}", font=self.font, fill=(255, 255, 255))
            draw.text((10, 56), f"Total Reward: {total_reward:.2f}", font=self.font, fill=(255, 255, 255))
            frames.append(np.array(img))
            
            action = self.agent.select_action(state, info)
            state, reward, done, info = self.env.step(action.item())
            total_reward += reward
        imageio.mimsave(self.filename, frames, duration=20)

    def record_milestone(self, milestone, episode):
        """
        Record a video of a milestone achievement.
        """
        filename = f"videos/milestone_{milestone}_{episode}.gif"
        self.record(episode)

    def record_comparison(self, agents, episode, filename='comparison.gif'):
        """
        Record a video comparing multiple agents playing the same episode.
        """
        frames = []
        for agent in agents:
            state = self.env.reset()
            done = False
            info = {'status': 'small'}
            while not done:
                frame = self.env.render(mode='rgb_array')
                img = Image.fromarray(frame)
                draw = ImageDraw.Draw(img)
                draw.text((10, 36), f"Agent: {agent.model.__class__.__name__}", font=self.font, fill=(255, 255, 255))
                frames.append(np.array(img))
                
                action = agent.select_action(state, info)
                state, _, done, info = self.env.step(action.item())
        imageio.mimsave(filename, frames, duration=20)
