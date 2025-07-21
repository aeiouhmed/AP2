import torch
import random
import numpy as np

class BaseAgent:
    """
    A base class for DQN agents.
    """
    def __init__(self, env, replay_buffer, model, target_model, optimizer, loss_fn, gamma, epsilon_start, epsilon_end, epsilon_decay):
        """
        Initialize the agent.
        """
        self.env = env
        self.replay_buffer = replay_buffer
        self.model = model
        self.target_model = target_model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

    def select_action(self, state, info):
        """
        Select an action using an epsilon-greedy policy.
        """
        sample = random.random()
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            np.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                if isinstance(state, np.ndarray):
                    state_tensor = torch.from_numpy(state).float().div(255.0).unsqueeze(0)
                else:
                    state_tensor = state.unsqueeze(0)
                q_values = self.model(state_tensor)
                # If not in fireball state, prevent fireball actions
                if info['status'] != 'fireball':
                    q_values[0][3] = -1e8  # Fireball right
                    q_values[0][8] = -1e8  # Fireball left
                return q_values.max(1)[1].view(1, 1)
        else:
            action = random.randrange(self.env.action_space.n)
            # If not in fireball state, prevent fireball actions
            if info['status'] != 'fireball' and action in [3, 8]:
                return torch.tensor([[0]], dtype=torch.long) # Default to no-op
            return torch.tensor([[action]], dtype=torch.long)

    def train(self, batch_size):
        """
        Train the agent.
        """
        raise NotImplementedError

    def update_target_model(self):
        """
        Update the target model with the weights of the online model.
        """
        self.target_model.load_state_dict(self.model.state_dict())
