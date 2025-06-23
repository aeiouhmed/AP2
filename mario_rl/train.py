import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import torch
import argparse
import numpy as np
from .environment.wrappers import GrayScaleObservation, ResizeObservation, FrameStack, SkipFrame
from .environment.reward_shaping import RewardShaper
from .models.replay_buffer import ReplayBuffer
from .models.networks import BaseCNN, DuelingDQN
from .agents.dqn_agent import DQNAgent
from .agents.ddqn_agent import DDQNAgent
from .agents.dueling_agent import create_dueling_agent
from .utils.logger import Logger
from .utils.genetic_algorithm import GeneticAlgorithm
from .utils.behavior_tracker import BehaviorTracker
from .utils.visualizer import Visualizer

import random
import os

def train(args, hyperparameters=None):
    """
    Train the agent.
    """
    if hyperparameters:
        args.lr = hyperparameters['lr']
        args.gamma = hyperparameters['gamma']
        args.epsilon_decay = hyperparameters['epsilon_decay']
    
    # Create recordings and models directories
    if not os.path.exists(f'recordings/{args.agent_name}'):
        os.makedirs(f'recordings/{args.agent_name}')
    if not os.path.exists('models'):
        os.makedirs('models')

    # Initialize logger and behavior tracker
    logger = Logger(f'runs/{args.agent}_{"dueling" if args.dueling else ""}_{args.lr}_{args.gamma}_{args.epsilon_decay}')
    behavior_tracker = BehaviorTracker()
    
    # Create environment
    env = gym.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)
    env = RewardShaper(env)

    # Create replay buffer
    replay_buffer = ReplayBuffer(args.buffer_size)
    
    # Create agent
    if args.dueling:
        model = DuelingDQN(env.observation_space.shape, env.action_space.n)
        target_model = DuelingDQN(env.observation_space.shape, env.action_space.n)
        target_model.load_state_dict(model.state_dict())
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        agent = create_dueling_agent(
            args.agent,
            env,
            replay_buffer,
            model,
            target_model,
            optimizer,
            torch.nn.MSELoss(),
            args.gamma,
            args.epsilon_start,
            args.epsilon_end,
            args.epsilon_decay
        )
    else:
        model = BaseCNN(env.observation_space.shape, env.action_space.n)
        target_model = BaseCNN(env.observation_space.shape, env.action_space.n)
        target_model.load_state_dict(model.state_dict())
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        if args.agent == 'dqn':
            agent = DQNAgent(
                env,
                replay_buffer,
                model,
                target_model,
                optimizer,
                torch.nn.MSELoss(),
                args.gamma,
                args.epsilon_start,
                args.epsilon_end,
                args.epsilon_decay
            )
        elif args.agent == 'ddqn':
            agent = DDQNAgent(
                env,
                replay_buffer,
                model,
                target_model,
                optimizer,
                torch.nn.MSELoss(),
                args.gamma,
                args.epsilon_start,
                args.epsilon_end,
                args.epsilon_decay
            )

    # Training loop
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32).div(255)
    episode_reward = 0
    episode_length = 0
    episode = 0
    total_rewards = []
    info = {'status': 'small'}
    for step in range(args.total_steps):
        # Select and perform an action
        action = agent.select_action(state, info)
        next_state, reward, done, info = env.step(action.item())
        
        # Update behavior tracker
        behavior_tracker.update(info, episode)
        
        # Store the transition in memory
        agent.replay_buffer.push(state, action, reward, next_state, done)
        
        # Move to the next state
        state = torch.tensor(next_state, dtype=torch.float32).div(255)
        episode_reward += reward
        episode_length += 1
        
        # If the episode is done, reset the environment
        if done:
            episode += 1
            behavior_tracker.check_level_completion(info, episode)
            logger.log_episode(episode, episode_reward, episode_length, step, info)
            total_rewards.append(episode_reward)
            state = env.reset()
            state = torch.tensor(state, dtype=torch.float32).div(255)
            episode_reward = 0
            episode_length = 0

        # Perform one step of the optimization
        loss, q_value = agent.train(args.batch_size)
        if loss is not None:
            logger.log_training(loss, q_value, step)

        # Update the target network
        if step % args.target_update == 0:
            agent.update_target_model()

        # Save the model
        if step % args.save_interval == 0:
            torch.save(agent.model.state_dict(), f'models/{args.agent}_{"dueling" if args.dueling else ""}_{step}.pth')

        # Visualize the agent's gameplay
        # if episode <= 50:
        #     env.render()
        # if episode > 0 and episode % 50 == 0:
        #     visualizer = Visualizer(env, agent, f'recordings/{args.agent_name}/{args.agent}_{"dueling" if args.dueling else ""}_{episode}.gif')
        #     visualizer.record(episode, episode_reward)

        # Evaluate the agent
        if step % args.eval_interval == 0:
            eval_reward = evaluate(args, agent, env)
            logger.log_scalar('eval/reward', eval_reward, step)
    
    # Close the logger
    logger.close()
    
    # Return the average reward over the last 100 episodes
    return np.mean(total_rewards[-100:])

def evaluate(args, agent, env):
    """
    Evaluate the agent's performance.
    """
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32).div(255)
    done = False
    total_reward = 0
    while not done:
        with torch.no_grad():
            action = agent.model(state.unsqueeze(0)).max(1)[1].view(1, 1)
        state, reward, done, _ = env.step(action.item())
        state = torch.tensor(state, dtype=torch.float32).div(255)
        total_reward += reward
        if done:
            state = env.reset()
            state = torch.tensor(state, dtype=torch.float32).div(255)
    return total_reward

def main(args):
    """
    Main function.
    """
    if args.use_ga:
        # Define the gene pool for the genetic algorithm
        gene_pool = {
            'lr': [1e-3, 1e-4, 1e-5],
            'gamma': [0.9, 0.99, 0.999],
            'epsilon_decay': [10000, 50000, 100000]
        }
        
        # Create the genetic algorithm
        ga = GeneticAlgorithm(population_size=10, gene_pool=gene_pool, fitness_fn=lambda hp: train(args, hp))
        
        # Evolve the population
        for generation in range(args.ga_generations):
            print(f"Generation {generation + 1}")
            ga.evolve()
            print(f"Best hyperparameters: {ga.population[0]}")
        
        # Train with the best hyperparameters
        best_hyperparameters = ga.population[0]
        train(args, best_hyperparameters)
    else:
        # Train with the default hyperparameters
        train(args)

from .configs.dqn_config import DQNConfig
from .configs.ddqn_config import DDQNConfig

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, default='dqn', help='Agent type (dqn, ddqn)')
    args, unknown = parser.parse_known_args()

    if args.agent == 'dqn':
        config = DQNConfig
    elif args.agent == 'ddqn':
        config = DDQNConfig
    
    parser.add_argument('--dueling', action='store_true', default=config.DUELING, help='Use Dueling DQN')
    parser.add_argument('--use_ga', action='store_true', help='Use Genetic Algorithm for hyperparameter tuning')
    parser.add_argument('--ga_generations', type=int, default=10, help='Number of generations for GA')
    parser.add_argument('--buffer_size', type=int, default=config.BUFFER_SIZE, help='Replay buffer size')
    parser.add_argument('--lr', type=float, default=config.LR, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=config.GAMMA, help='Discount factor')
    parser.add_argument('--epsilon_start', type=float, default=config.EPSILON_START, help='Epsilon start value')
    parser.add_argument('--epsilon_end', type=float, default=config.EPSILON_END, help='Epsilon end value')
    parser.add_argument('--epsilon_decay', type=int, default=config.EPSILON_DECAY, help='Epsilon decay rate')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE, help='Batch size')
    parser.add_argument('--total_steps', type=int, default=config.TOTAL_STEPS, help='Total training steps')
    parser.add_argument('--target_update', type=int, default=config.TARGET_UPDATE, help='Target network update frequency')
    parser.add_argument('--save_interval', type=int, default=config.SAVE_INTERVAL, help='Model save interval')
    parser.add_argument('--vis_interval', type=int, default=config.VIS_INTERVAL, help='Visualization interval')
    parser.add_argument('--eval_interval', type=int, default=config.EVAL_INTERVAL, help='Evaluation interval')
    parser.add_argument('--agent_name', type=str, default='mario', help='Name of the agent')
    args = parser.parse_args()
    main(args)
