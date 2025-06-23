import torch
from .base_agent import BaseAgent

class DDQNAgent(BaseAgent):
    """
    A Double DQN agent.
    """
    def train(self, batch_size):
        """
        Train the agent for one step.
        """
        if len(self.replay_buffer) < batch_size:
            return None, None
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Get current Q values
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Get next actions from online network
        next_actions = self.model(next_states).max(1)[1].unsqueeze(1)
        # Get next Q values from target network
        next_q_values = self.target_model(next_states).gather(1, next_actions).squeeze(1)
        
        # Compute the expected Q values
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute loss
        loss = self.loss_fn(q_values, expected_q_values.detach())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), q_values.mean().item()
