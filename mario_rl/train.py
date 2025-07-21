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
from .environment.wrappers import ProlongedJumpWrapper
from .utils.genetic_algorithm import GeneticAlgorithm
from .utils.behavior_tracker import BehaviorTracker
from .utils.visualizer import Visualizer

import random
import os
import sys
import json
from datetime import datetime

def train(args, hyperparameters=None):
    # Create a log file
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"{args.agent_label}_{datetime.now().strftime('%Y%m%d%H%M%S')}.txt")
    
    class LoggerTee:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()

    log_file = open(log_file_path, 'w')
    original_stdout = sys.stdout
    sys.stdout = LoggerTee(sys.stdout, log_file)
    """
    Train the agent.
    """
    if hyperparameters:
        args.lr = hyperparameters['lr']
        args.gamma = hyperparameters['gamma']
        args.epsilon_decay = hyperparameters['epsilon_decay']
    
    # Create recordings and models directories
    if not os.path.exists(f'recordings/{args.agent_label}'):
        os.makedirs(f'recordings/{args.agent_label}')
    if not os.path.exists('models'):
        os.makedirs('models')

    # Initialize logger and behavior tracker
    logger = Logger(f'runs/{args.agent_label}')
    behavior_tracker = BehaviorTracker()
    
    # Create environment
    env = gym.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = ProlongedJumpWrapper(env)
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
        if args.load_model:
            model.load_state_dict(torch.load(args.load_model))
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
        if args.load_model:
            model.load_state_dict(torch.load(args.load_model))
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
    step = 0
    while True:
        step += 1
        if step > args.total_steps:
            break

        # Select and perform an action
        action = agent.select_action(state, info)
        next_state, reward, done, info = env.step(action.item())
        
        # Update behavior tracker
        behavior_tracker.update(info, episode)
        
        # Store the transition in memory
        agent.replay_buffer.push(state, action, reward, next_state, done)
        
        episode_reward += reward
        episode_length += 1
        
        # Print metrics every 500 steps
        if step % 500 == 0:
            print(f"Step: {step}, Episode: {episode}, Stage: {info['world']}-{info['stage']}, Score: {info['score']}, Time: {info['time']}, Episode Max X: {behavior_tracker.episode_max_x_pos}, All-Time Max X: {behavior_tracker.milestones['max_x_pos']}, Episode Reward: {episode_reward}")

        # If the episode is done, reset the environment
        if done:
            episode += 1
            behavior_tracker.check_level_completion(info, episode)
            behavior_tracker.reset_episode_stats()
            logger.log_episode(episode, episode_reward, episode_length, step, info)
            logger.log_milestones(behavior_tracker.milestones, episode)
            total_rewards.append(episode_reward)

            # Visualize the agent's gameplay
            if episode > 0 and episode % args.rec_frequency == 0:
                timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                visualizer = Visualizer(agent, f'recordings/{args.agent_label}/{episode}_{timestamp}.gif')
                visualizer.record(episode)

            state = env.reset()
            state = torch.tensor(state, dtype=torch.float32).div(255)
            episode_reward = 0
            episode_length = 0
        else:
            # Move to the next state
            state = torch.tensor(next_state, dtype=torch.float32).div(255)

        # Perform one step of the optimization
        loss, q_value = agent.train(args.batch_size)
        if loss is not None:
            logger.log_training(loss, q_value, step)

        # Update the target network
        if step % args.target_update == 0:
            agent.update_target_model()

        # Save the model
        if step % args.save_interval == 0:
            torch.save(agent.model.state_dict(), f'models/{args.agent_label}_{step}.pth')

        # Render the environment
        if args.render:
            env.render()

        # Evaluate the agent
        if step % args.eval_interval == 0:
            eval_reward = evaluate(args, agent, env)
            logger.log_scalar('eval/reward', eval_reward, step)
    
    # Close the logger and log file
    logger.close()
    log_file.close()
    sys.stdout = original_stdout
    
    # Return the agent and the average reward over the last 100 episodes
    return agent, np.mean(total_rewards[-100:])

def evaluate(args, agent, env, eval_episodes=10):
    """
    Evaluate the agent's performance over a number of episodes.
    """
    agent.model.eval()  # Set the model to evaluation mode
    total_rewards = []
    for episode in range(eval_episodes):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32).div(255)
        done = False
        episode_reward = 0
        while not done:
            with torch.no_grad():
                # Use a small epsilon for exploration during evaluation
                if np.random.random() < 0.05:
                    action = env.action_space.sample()
                else:
                    q_values = agent.model(state.unsqueeze(0))
                    action = q_values.max(1)[1].item()
            
            next_state, reward, done, _ = env.step(action)
            state = torch.tensor(next_state, dtype=torch.float32).div(255)
            episode_reward += reward
        total_rewards.append(episode_reward)
    
    agent.model.train() # Set the model back to training mode
    return np.mean(total_rewards)

def main(args):
    """
    Main function.
    """
    # If total_steps is provided as a command-line argument,
    # override epsilon_decay and eval_interval
    if args.total_steps != config.TOTAL_STEPS:
        args.epsilon_decay = args.total_steps
        args.eval_interval = args.total_steps -1


    if args.find_hyperparameters:
        # Hyperparameter search mode
        print("--- Starting Hyperparameter Search ---")
        gene_pool = {
            'lr': [1e-3, 1e-4, 1e-5],
            'gamma': [0.9, 0.99, 0.999],
            'epsilon_decay': [10000, 50000, 100000]
        }
        
        ga_args = argparse.Namespace(**vars(args))
        ga_args.total_steps = ga_args.ga_steps
        ga = GeneticAlgorithm(population_size=10, gene_pool=gene_pool, fitness_fn=lambda hp: train(ga_args, hp)[1])
        
        hyperparameters_dir = 'hyperparameters'
        os.makedirs(hyperparameters_dir, exist_ok=True)

        for generation in range(args.ga_generations):
            print(f"Generation {generation + 1}/{args.ga_generations}")
            ga.evolve()
            best_hyperparameters = ga.population[0]
            print(f"Best hyperparameters so far: {best_hyperparameters}")

            # Save the best hyperparameters after each generation
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            hyperparameter_file_path = os.path.join(hyperparameters_dir, f"hp_{timestamp}.json")
            
            with open(hyperparameter_file_path, 'w') as f:
                json.dump(best_hyperparameters, f, indent=4)
            print(f"Best hyperparameters for generation {generation + 1} saved to {hyperparameter_file_path}")

        print(f"--- Hyperparameter Search Complete ---")

    else:
        # Agent training mode
        print("--- Starting Agent Training ---")
        hyperparameters = None
        if args.hyperparameter_file:
            with open(args.hyperparameter_file, 'r') as f:
                hyperparameters = json.load(f)
            print(f"Loaded hyperparameters from {args.hyperparameter_file}")
        
        agent, _ = train(args, hyperparameters)
        
        # Save the final model
        final_model_path = f'models/{args.agent_label}_final.pth'
        torch.save(agent.model.state_dict(), final_model_path)
        print(f"Final model saved to {final_model_path}")
        
        print("--- Agent Training Complete ---")
        
        # Evaluate the final model
        print("--- Starting Evaluation ---")
        # We need to create a new env for evaluation because the training env is closed.
        eval_env = gym.make('SuperMarioBros-v0')
        eval_env = JoypadSpace(eval_env, COMPLEX_MOVEMENT)
        eval_env = ProlongedJumpWrapper(eval_env)
        eval_env = SkipFrame(eval_env, skip=4)
        eval_env = GrayScaleObservation(eval_env)
        eval_env = ResizeObservation(eval_env, shape=84)
        eval_env = FrameStack(eval_env, num_stack=4)
        eval_env = RewardShaper(eval_env)
        evaluate(args, agent, eval_env, eval_episodes=args.eval_episodes)
        eval_env.close()
        print("--- Evaluation Complete ---")

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
    
    # Arguments for hyperparameter search
    parser.add_argument('--find_hyperparameters', action='store_true', help='Run in hyperparameter search mode')
    parser.add_argument('--ga_generations', type=int, default=5, help='Number of generations for GA')
    parser.add_argument('--ga_steps', type=int, default=10000, help='Number of steps for each GA evaluation')

    # Arguments for agent training
    parser.add_argument('--hyperparameter_file', type=str, default=None, help='Path to a JSON file with hyperparameters')
    parser.add_argument('--total_steps', type=int, default=config.TOTAL_STEPS, help='Total training steps')

    # General arguments
    parser.add_argument('--buffer_size', type=int, default=config.BUFFER_SIZE, help='Replay buffer size')
    parser.add_argument('--lr', type=float, default=config.LR, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=config.GAMMA, help='Discount factor')
    parser.add_argument('--epsilon_start', type=float, default=config.EPSILON_START, help='Epsilon start value')
    parser.add_argument('--epsilon_end', type=float, default=config.EPSILON_END, help='Epsilon end value')
    parser.add_argument('--epsilon_decay', type=int, default=config.EPSILON_DECAY, help='Epsilon decay rate')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE, help='Batch size')
    parser.add_argument('--target_update', type=int, default=config.TARGET_UPDATE, help='Target network update frequency')
    parser.add_argument('--save_interval', type=int, default=config.SAVE_INTERVAL, help='Model save interval')
    parser.add_argument('--vis_interval', type=int, default=config.VIS_INTERVAL, help='Visualization interval')
    parser.add_argument('--eval_interval', type=int, default=config.EVAL_INTERVAL, help='Evaluation interval')
    parser.add_argument('--eval_episodes', type=int, default=10, help='Number of episodes for evaluation')
    parser.add_argument('--agent_label', type=str, default='mario', help='A label for the agent')
    parser.add_argument('--load_model', type=str, default=None, help='Path to a saved model to continue training from')
    parser.add_argument('--rec-frequency', type=int, default=50, help='Frequency of recording episodes')
    parser.add_argument('--render', action='store_true', help='Render the environment during training')
    args = parser.parse_args()
    main(args)
