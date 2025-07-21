import gym
import numpy as np

class RewardShaper(gym.Wrapper):
    """
    Shape the reward function to provide more meaningful signals to the agent.
    """
    def __init__(self, env):
        super().__init__(env)
        self.previous_x_pos = 0
        self.max_x_pos = 0
        self.previous_coins = 0
        self.previous_life = 2
        self.previous_score = 0
        self.previous_status = 'small'
        self.x_pos_history = []
        self.stuck_counter = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
        x_pos_diff = info['x_pos'] - self.previous_x_pos

        # Reward for making progress to the right (reaching a new max x-position)
        if info['x_pos'] > self.max_x_pos:
            reward += (info['x_pos'] - self.max_x_pos) * 60.0
            self.max_x_pos = info['x_pos']
        # Penalty for moving left (losing progress)
        elif x_pos_diff < 0:
            reward += x_pos_diff * 60.0

        # Check if agent is stuck
        is_moving_right = action in [1, 2, 3, 4]
        if x_pos_diff <= 0 and is_moving_right:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

        # Reward for jumping when stuck
        if self.stuck_counter > 10:  # Threshold for being stuck
            is_jumping_right = action in [2, 4]
            is_moving_left = action == 6
            if is_jumping_right:
                reward += 2500
            elif is_moving_left:
                reward += 2480
        
        self.previous_x_pos = info['x_pos']
        
        # Penalty for idling
        self.x_pos_history.append(info['x_pos'])
        if len(self.x_pos_history) > 100:
            self.x_pos_history.pop(0)
            if np.all(np.array(self.x_pos_history) == self.x_pos_history[0]):
                reward -= 1000

        # Reward for collecting coins
        reward += (info['coins'] - self.previous_coins) * 10
        self.previous_coins = info['coins']
        
        # Reward for defeating enemies and getting mushrooms
        score_diff = info['score'] - self.previous_score
        if score_diff > 0:
            reward += score_diff * 10.0
        self.previous_score = info['score']
        
        # Reward for getting power-ups
        if info['status'] == 'tall' and self.previous_status == 'small':
            reward += 100
        if info['status'] == 'fireball' and self.previous_status == 'tall':
            reward += 200
        self.previous_status = info['status']
        
        # Penalty for losing a life
        if info['life'] < self.previous_life:
            reward -= 5000
        self.previous_life = info['life']
        
        # Penalty for dying
        if done and info['life'] == 0:
            reward -= 5000
            
        # Reward for completing the level
        if info['flag_get']:
            reward += 50000
            
        # Penalty for timer reaching zero
        if info['time'] == 0:
            reward -= 1500
            done = True
            
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        self.previous_x_pos = 0
        self.max_x_pos = 0
        self.previous_coins = 0
        self.previous_life = 2
        self.previous_score = 0
        self.previous_status = 'small'
        self.x_pos_history = []
        self.stuck_counter = 0
        return obs
