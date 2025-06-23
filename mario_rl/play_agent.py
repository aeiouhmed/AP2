import gym
import torch
import argparse
import keyboard
from .environment.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
from .models.networks import BaseCNN, DuelingDQN

def main(args):
    if args.dueling:
        model = DuelingDQN(gym.make('SuperMarioBros-v0').observation_space.shape, gym.make('SuperMarioBros-v0').action_space.n)
    else:
        model = BaseCNN(gym.make('SuperMarioBros-v0').observation_space.shape, gym.make('SuperMarioBros-v0').action_space.n)
    
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    for world in range(1, 9):
        for stage in range(1, 5):
            try:
                env = gym.make(f'SuperMarioBros-{world}-{stage}-v0')
                env = GrayScaleObservation(env)
                env = ResizeObservation(env, shape=84)
                env = FrameStack(env, num_stack=4)

                state = env.reset()
                state = torch.tensor(state, dtype=torch.float32).div(255)
                done = False
                total_reward = 0
                while not done:
                    if args.human_takeover and keyboard.is_pressed('h'):
                        action = env.action_space.sample()
                    else:
                        with torch.no_grad():
                            action = model(state.unsqueeze(0)).max(1)[1].view(1, 1)
                    
                    state, reward, done, _ = env.step(action.item())
                    state = torch.tensor(state, dtype=torch.float32).div(255)
                    env.render()
                    total_reward += reward
                
                print(f"World {world}-{stage}: Total reward = {total_reward}")
                env.close()
            except Exception as e:
                print(f"Error loading level {world}-{stage}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--dueling', action='store_true', help='Use Dueling DQN')
    parser.add_argument('--human_takeover', action='store_true', help='Allow human takeover')
    args = parser.parse_args()
    main(args)
