import gym
import torch
import argparse
import keyboard
import time
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from .environment.wrappers import GrayScaleObservation, ResizeObservation, FrameStack, SkipFrame, ProlongedJumpWrapper
from .environment.reward_shaping import RewardShaper
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
    # Create a dummy environment to get the correct observation shape
    dummy_env = gym.make('SuperMarioBros-v0')
    dummy_env = JoypadSpace(dummy_env, COMPLEX_MOVEMENT)
    dummy_env = ProlongedJumpWrapper(dummy_env)
    dummy_env = SkipFrame(dummy_env, skip=4)
    dummy_env = GrayScaleObservation(dummy_env)
    dummy_env = ResizeObservation(dummy_env, shape=84)
    dummy_env = FrameStack(dummy_env, num_stack=4)
    
    if args.dueling:
        model = DuelingDQN(dummy_env.observation_space.shape, len(COMPLEX_MOVEMENT))
    else:
        model = BaseCNN(dummy_env.observation_space.shape, len(COMPLEX_MOVEMENT))
    
    dummy_env.close()
    
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    for world in range(1, 9):
        for stage in range(1, 5):
            try:
                env = gym.make(f'SuperMarioBros-{world}-{stage}-v0')
                env = JoypadSpace(env, COMPLEX_MOVEMENT)
                env = ProlongedJumpWrapper(env)
                env = SkipFrame(env, skip=4)
                env = GrayScaleObservation(env)
                env = ResizeObservation(env, shape=84)
                env = FrameStack(env, num_stack=4)
                env = RewardShaper(env)

                state = env.reset()
                state = torch.tensor(state, dtype=torch.float32).div(255)
                done = False
                total_reward = 0
                human_control = False
                last_toggle_time = 0
                while not done:
                    if args.human_takeover and keyboard.is_pressed('h') and (time.time() - last_toggle_time > 0.5):
                        human_control = not human_control
                        print(f"Human control: {human_control}")
                        last_toggle_time = time.time()
                    
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
