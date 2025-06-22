import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import keyboard

def main():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    print("Human control enabled. Use arrow keys to move, 'a' to jump, 's' to run.")
    print("Press 'q' to quit.")

    done = True
    while True:
        if done:
            state = env.reset()
        
        action = 0  # Default action: NOOP
        if keyboard.is_pressed('right'):
            action = 5  # Right
        elif keyboard.is_pressed('left'):
            action = 1  # Left
        
        if keyboard.is_pressed('a'):
            if keyboard.is_pressed('right'):
                action = 6  # Right + Jump
            elif keyboard.is_pressed('left'):
                action = 2  # Left + Jump
            else:
                action = 3 # Jump
        
        if keyboard.is_pressed('s'):
            if keyboard.is_pressed('right'):
                action = 7 # Right + Run
            elif keyboard.is_pressed('left'):
                action = 4 # Left + Run

        state, reward, done, info = env.step(action)
        env.render()

        if keyboard.is_pressed('q'):
            break

    env.close()

if __name__ == "__main__":
    main()
