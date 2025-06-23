import gym
import torch
import argparse
import keyboard
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from .environment.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
from .models.networks import BaseCNN, DuelingDQN

def get_human_action():
    action = 0  # Default action: NOOP
    if keyboard.is_pressed('d'):
        action = 1  # Right
    elif keyboard.is_pressed('a'):
        action = 6  # Left
    
    if keyboard.is_pressed('w'):
        if keyboard.is_pressed('d'):
            action = 2 # Right + Jump
        elif keyboard.is_pressed('a'):
            action = 7 # Left + Jump
        else:
            action = 5 # Jump
    return action

def main(args):
    if args.dueling:
        model = DuelingDQN(gym.make('SuperMarioBros-v0').observation_space.shape, len(COMPLEX_MOVEMENT))
    else:
        model = BaseCNN(gym.make('SuperMarioBros-v0').observation_space.shape, len(COMPLEX_MOVEMENT))
    
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    for world in range(1, 9):
        for stage in range(1, 5):
            try:
                env = gym.make(f'SuperMarioBros-{world}-{stage}-v0')
                env = JoypadSpace(env, COMPLEX_MOVEMENT)
                env = GrayScaleObservation(env)
                env = ResizeObservation(env, shape=84)
                env = FrameStack(env, num_stack=4)

                state = env.reset()
                state = torch.tensor(state, dtype=torch.float32).div(255)
                done = False
                total_reward = 0
                human_control = False
                while not done:
                    if args.human_takeover and keyboard.is_pressed('h'):
                        human_control = not human_control
                        print(f"Human control: {human_control}")
                    
                    if human_control:
                        action = get_human_action()
                    else:
                        with torch.no_grad():
                            action_tensor = model(state.unsqueeze(0)).max(1)[1].view(1, 1)
                            action = action_tensor.item()
                    
                    state, reward, done, _ = env.step(action)
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
