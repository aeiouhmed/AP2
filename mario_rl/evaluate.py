import gym
import torch
import argparse
import csv
import numpy as np
from captum.attr import Saliency
from .environment.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
from .models.networks import BaseCNN, DuelingDQN
import matplotlib.pyplot as plt

from .utils.visualizer import Visualizer

def main(args):
    agents = []
    for model_path in args.model_paths:
        if args.dueling:
            model = DuelingDQN(gym.make('SuperMarioBros-v0').observation_space.shape, gym.make('SuperMarioBros-v0').action_space.n)
        else:
            model = BaseCNN(gym.make('SuperMarioBros-v0').observation_space.shape, gym.make('SuperMarioBros-v0').action_space.n)
        
        model.load_state_dict(torch.load(model_path))
        model.eval()
        agents.append(model)

    if args.compare:
        env = gym.make('SuperMarioBros-1-1-v0')
        env = GrayScaleObservation(env)
        env = ResizeObservation(env, shape=84)
        env = FrameStack(env, num_stack=4)
        visualizer = Visualizer(env, None)
        visualizer.record_comparison(agents, 0)
        env.close()

    with open('evaluation_results.csv', 'w', newline='') as csvfile:
        fieldnames = ['algorithm', 'level', 'total_reward']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, agent in enumerate(agents):
            model_path = args.model_paths[i]
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
                            env.render()
                            with torch.no_grad():
                                input_tensor = state.unsqueeze(0)
                                action = agent(input_tensor).max(1)[1].view(1, 1)
                            
                            if args.saliency:
                                saliency = Saliency(agent)
                                grads = saliency.attribute(input_tensor, target=action.item())
                                saliency_map = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))
                                
                                plt.imshow(saliency_map)
                                plt.pause(0.01) # Pause to update the plot

                            next_state, reward, done, _ = env.step(action.item())
                            state = torch.tensor(next_state, dtype=torch.float32).div(255)
                            total_reward += reward
                        
                        writer.writerow({'algorithm': model_path, 'level': f'{world}-{stage}', 'total_reward': total_reward})
                        print(f"Algorithm: {model_path}, World {world}-{stage}: Total reward = {total_reward}")
                        env.close()
                    except Exception as e:
                        print(f"Error loading level {world}-{stage}: {e}")
                        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_paths', nargs='+', required=True, help='Paths to the trained models')
    parser.add_argument('--dueling', action='store_true', help='Use Dueling DQN')
    parser.add_argument('--saliency', action='store_true', help='Visualize saliency maps')
    parser.add_argument('--compare', action='store_true', help='Compare agents')
    args = parser.parse_args()
    main(args)
