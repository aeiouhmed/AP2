from torch.utils.tensorboard import SummaryWriter
import numpy as np

class Logger:
    """
    A simple logger that writes to a TensorBoard log file.
    """
    def __init__(self, log_dir):
        """
        Initialize the logger.
        Args:
            log_dir (str): The directory to save the log file.
        """
        self.writer = SummaryWriter(log_dir)
        self.episode_rewards = []

    def log_scalar(self, tag, value, step):
        """
        Log a scalar value.
        """
        self.writer.add_scalar(tag, value, step)

    def log_episode(self, episode, total_reward, episode_length, step, info):
        """
        Log episode metrics.
        """
        self.log_scalar('episode/total_reward', total_reward, episode)
        self.log_scalar('episode/episode_length', episode_length, episode)
        self.log_scalar('episode/steps', step, episode)
        self.log_scalar('episode/coins', info['coins'], episode)
        self.log_scalar('episode/score', info['score'], episode)
        self.log_scalar('episode/stage', info['stage'], episode)
        self.log_scalar('episode/world', info['world'], episode)
        self.log_scalar('episode/time', info['time'], episode)
        self.log_scalar('episode/x_pos', info['x_pos'], episode)
        self.log_scalar('episode/y_pos', info['y_pos'], episode)

        self.episode_rewards.append(total_reward)
        if len(self.episode_rewards) > 100:
            self.episode_rewards.pop(0)
        
        if len(self.episode_rewards) == 100:
            moving_avg_reward = np.mean(self.episode_rewards)
            self.log_scalar('episode/moving_avg_reward', moving_avg_reward, episode)

    def log_training(self, loss, q_value, step):
        """
        Log training metrics.
        """
        self.log_scalar('train/loss', loss, step)
        self.log_scalar('train/q_value', q_value, step)

    def close(self):
        """
        Close the logger.
        """
        self.writer.close()
