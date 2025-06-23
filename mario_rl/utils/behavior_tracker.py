import numpy as np

class BehaviorTracker:
    """
    A class for tracking the agent's behavior and learning milestones.
    """
    def __init__(self, visualizer=None):
        self.milestones = {
            'max_x_pos': 0,
            'enemies_defeated': 0,
            'coins_collected': 0,
            'levels_completed': 0,
        }
        self.x_pos_history = []
        self.visualizer = visualizer

    def update(self, info, episode):
        """
        Update the behavior tracker with the latest info from the environment.
        """
        if info['x_pos'] > self.milestones['max_x_pos']:
            self.milestones['max_x_pos'] = info['x_pos']
            print(f"New milestone: Reached x-position {info['x_pos']}")
            if self.visualizer:
                self.visualizer.record_milestone('max_x_pos', episode)

        self.x_pos_history.append(info['x_pos'])
        if len(self.x_pos_history) > 100:
            self.x_pos_history.pop(0)
        
        self.detect_waiting()
        self.detect_backtracking()

        # These are placeholders, as the environment doesn't provide this info directly.
        # We would need to implement a custom reward wrapper to track these metrics.
        # if info['enemies_defeated'] > self.milestones['enemies_defeated']:
        #     self.milestones['enemies_defeated'] = info['enemies_defeated']
        #     print(f"New milestone: Defeated {info['enemies_defeated']} enemies")
            
        # if info['coins'] > self.milestones['coins_collected']:
        #     self.milestones['coins_collected'] = info['coins']
        #     print(f"New milestone: Collected {info['coins']} coins")

    def check_level_completion(self, info, episode):
        """
        Check if the agent has completed the level.
        """
        if info['flag_get']:
            self.milestones['levels_completed'] += 1
            print(f"New milestone: Completed level {self.milestones['levels_completed']}")
            if self.visualizer:
                self.visualizer.record_milestone('level_completion', episode)

    def detect_waiting(self):
        """
        Detect if the agent is waiting for an enemy to pass.
        """
        if len(self.x_pos_history) == 100:
            if np.all(np.array(self.x_pos_history) == self.x_pos_history[0]):
                print("Strategic behavior detected: Waiting")

    def detect_backtracking(self):
        """
        Detect if the agent is backtracking.
        """
        if len(self.x_pos_history) == 100:
            if self.x_pos_history[-1] < self.x_pos_history[0]:
                print("Strategic behavior detected: Backtracking")
