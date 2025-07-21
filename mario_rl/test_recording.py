import gym
import torch
import argparse
from .environment.wrappers import GrayScaleObservation, ResizeObservation, FrameStack, SkipFrame
from .environment.reward_shaping import RewardShaper
from .models.networks import BaseCNN, DuelingDQN
from .agents.dqn_agent import DQNAgent
from .agents.ddqn_agent import DDQNAgent
from .agents.dueling_agent import create_dueling_agent
from .utils.visualizer import Visualizer
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

def test_recording(args):
    """
    Test the recording functionality.
    """
    # Create environment
    env = gym.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)
    env = RewardShaper(env)

    # Create agent
    if args.dueling:
        model = DuelingDQN(env.observation_space.shape, env.action_space.n)
        agent = create_dueling_agent(args.agent, env, None, model, None, None, None, 0, 0, 0, 1)
    else:
        model = BaseCNN(env.observation_space.shape, env.action_space.n)
        if args.agent == 'dqn':
            agent = DQNAgent(env, None, model, None, None, None, 0, 0, 0, 1)
        elif args.agent == 'ddqn':
            agent = DDQNAgent(env, None, model, None, None, None, 0, 0, 0, 1)

    if args.load_model:
        model.load_state_dict(torch.load(args.load_model))
    
    # Create visualizer and record
    visualizer = Visualizer(env, agent, f'recordings/{args.agent_label}/test_recording.gif')
    visualizer.record(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, default='dqn', help='Agent type (dqn, ddqn)')
    parser.add_argument('--dueling', action='store_true', help='Use Dueling DQN')
    parser.add_argument('--agent_label', type=str, default='mario', help='A label for the agent')
    parser.add_argument('--load_model', type=str, default='models/dqn__0.pth', help='Path to a saved model to test')
    args = parser.parse_args()
    test_recording(args)
