import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class Visualizer:
    """
    A class for visualizing the agent's gameplay.
    """
    def __init__(self, env, agent, filename='play.gif'):
        self.env = env
        self.agent = agent
        self.filename = filename
        try:
            self.font = ImageFont.truetype("arial.ttf", 15)
        except IOError:
            self.font = ImageFont.load_default()

    def record(self, episode, total_reward):
        """
        Record a video of the agent playing one episode.
        """
        frames = []
        state = self.env.reset()
        done = False
        while not done:
            frame = self.env.render(mode='rgb_array')
            img = Image.fromarray(frame)
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), f"Episode: {episode}", font=self.font, fill=(255, 255, 255))
            draw.text((10, 30), f"Total Reward: {total_reward:.2f}", font=self.font, fill=(255, 255, 255))
            frames.append(np.array(img))
            
            action = self.agent.select_action(state)
            state, _, done, _ = self.env.step(action.item())
        imageio.mimsave(self.filename, frames, fps=30)

    def record_milestone(self, milestone, episode):
        """
        Record a video of a milestone achievement.
        """
        filename = f"videos/milestone_{milestone}_{episode}.gif"
        self.record(episode, 0)

    def record_comparison(self, agents, episode, filename='comparison.gif'):
        """
        Record a video comparing multiple agents playing the same episode.
        """
        frames = []
        for agent in agents:
            state = self.env.reset()
            done = False
            while not done:
                frame = self.env.render(mode='rgb_array')
                img = Image.fromarray(frame)
                draw = ImageDraw.Draw(img)
                draw.text((10, 10), f"Agent: {agent.model.__class__.__name__}", font=self.font, fill=(255, 255, 255))
                frames.append(np.array(img))
                
                action = agent.select_action(state)
                state, _, done, _ = self.env.step(action.item())
        imageio.mimsave(filename, frames, fps=30)
