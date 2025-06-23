from .dqn_agent import DQNAgent
from .ddqn_agent import DDQNAgent
from ..models.networks import DuelingDQN

def create_dueling_agent(agent_type, env, replay_buffer, model, target_model, optimizer, loss_fn, gamma, epsilon_start, epsilon_end, epsilon_decay):
    """
    Factory function for creating a Dueling DQN agent.
    """
    if agent_type == 'dqn':
        return DQNAgent(
            env,
            replay_buffer,
            model,
            target_model,
            optimizer,
            loss_fn,
            gamma,
            epsilon_start,
            epsilon_end,
            epsilon_decay
        )
    elif agent_type == 'ddqn':
        return DDQNAgent(
            env,
            replay_buffer,
            model,
            target_model,
            optimizer,
            loss_fn,
            gamma,
            epsilon_start,
            epsilon_end,
            epsilon_decay
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
