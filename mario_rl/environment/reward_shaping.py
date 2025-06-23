import gym

class RewardShaper(gym.Wrapper):
    """
    Shape the reward function to provide more meaningful signals to the agent.
    """
    def __init__(self, env):
        super().__init__(env)
        self.previous_x_pos = 0
        self.previous_coins = 0
        self.previous_life = 2
        self.previous_score = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
        # Reward for moving right
        reward += (info['x_pos'] - self.previous_x_pos) * 0.1
        self.previous_x_pos = info['x_pos']
        
        # Reward for collecting coins
        reward += (info['coins'] - self.previous_coins) * 10
        self.previous_coins = info['coins']
        
        # Reward for defeating enemies and getting mushrooms
        score_diff = info['score'] - self.previous_score
        if score_diff > 0:
            reward += score_diff * 0.1
        self.previous_score = info['score']
        
        # Penalty for losing a life
        if info['life'] < self.previous_life:
            reward -= 50
        self.previous_life = info['life']
        
        # Penalty for dying
        if done and info['life'] == 0:
            reward -= 100
            
        # Reward for completing the level
        if info['flag_get']:
            reward += 500
            
        # Penalty for taking too long
        reward -= 0.1
        
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        self.previous_x_pos = 0
        self.previous_coins = 0
        self.previous_life = 2
        self.previous_score = 0
        return obs
