import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import keyboard
import time

def main():
    env = gym.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)

    print("Human control enabled. Use WASD to move and jump.")
    print("Press 'q' to quit.")

    done = True
    while True:
        if done:
            state = env.reset()
        
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


        _, _, done, _ = env.step(action)
        env.render()
        time.sleep(0.01)

        if keyboard.is_pressed('q'):
            break

    env.close()

if __name__ == "__main__":
    main()
