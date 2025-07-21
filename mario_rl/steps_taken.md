# Steps Taken

1.  **Initial Planning:**
    *   Analyzed the user's request to train a DRL agent for Super Mario Bros.
    *   Developed a comprehensive project plan, including environment setup, agent architecture, training pipeline, and evaluation.
    *   Refined the plan based on user feedback to include DQN, DDQN, Dueling DQN, and extensive logging/visualization.
    *   Incorporated academic project requirements, including a detailed report structure and additional project objectives.

2.  **Project Scaffolding:**
    *   Created the main project directory `mario_rl`.
    *   Set up the subdirectory structure for agents, environment, models, utils, and configs.
    *   Created empty Python files for all planned modules.

3.  **Environment Setup and Debugging:**
    *   Implemented `play_human.py` to test the environment with human control.
    *   Encountered a persistent `OverflowError` in the `nes-py` library.
    *   Attempted multiple solutions to fix the error:
        *   Upgraded/downgraded `nes-py` and `gym-super-mario-bros`.
        *   Used a different ROM file.
        *   Explicitly loaded the ROM by its path.
        *   Registered the ROM as a custom Gym environment.
        *   Created a clean virtual environment.
        *   Patched the `nes_py/_rom.py` file.
    *   The user fixed the error in a separate task.
    *   Confirmed that the environment is now working correctly with `play_human.py`.

4.  **Environment Wrappers:**
    *   Installed `opencv-python` for image processing.
    *   Implemented `GrayScaleObservation`, `ResizeObservation`, and `FrameStack` wrappers in `mario_rl/environment/wrappers.py`.
    *   Implemented the `SkipFrame` wrapper.
    *   Implemented the `RewardShaper` wrapper.

5.  **Neural Network Models:**
    *   Installed `torch`, `torchvision`, and `torchaudio`.
    *   Implemented the `BaseCNN` and `DuelingDQN` models in `mario_rl/models/networks.py`.

6.  **Replay Buffer:**
    *   Implemented the `ReplayBuffer` class in `mario_rl/models/replay_buffer.py`.

7.  **Agent Implementation:**
    *   Implemented the `BaseAgent` class in `mario_rl/agents/base_agent.py`.
    *   Implemented the `DQNAgent` class in `mario_rl/agents/dqn_agent.py`.
    *   Implemented the `DDQNAgent` class in `mario_rl/agents/ddqn_agent.py`.
    *   Implemented the `DuelingAgent` factory in `mario_rl/agents/dueling_agent.py`.

8.  **Training Pipeline:**
    *   Implemented the main training script `train.py`.
    *   Added command-line arguments for hyperparameter configuration.
    *   Implemented the main training loop.
    *   Added model checkpointing.
    *   Added visualization scheduling.
    *   Added periodic evaluation.

9.  **Logging and Visualization:**
    *   Installed `tensorboard` and `imageio`.
    *   Implemented the `Logger` class in `mario_rl/utils/logger.py`.
    *   Implemented the `Visualizer` class in `mario_rl/utils/visualizer.py`.
    *   Integrated logging and visualization into the training pipeline.
    *   Enhanced the `Logger` to include rolling statistics.
    *   Enhanced the `Visualizer` to record milestone achievements and agent comparisons.

10. **Genetic Algorithm:**
    *   Implemented the `GeneticAlgorithm` class in `mario_rl/utils/genetic_algorithm.py`.
    *   Integrated the genetic algorithm into the training pipeline for hyperparameter optimization.

11. **Evaluation and Deployment:**
    *   Implemented the `evaluate.py` script for evaluating a trained model.
    *   Implemented the `play_agent.py` script for watching a trained agent play.
    *   Enhanced the `evaluate.py` script to run the agent through all levels, generate performance statistics, and compare different algorithms.
    *   Enhanced the `play_agent.py` script to run the agent through all levels and display performance metrics.
    *   Added human takeover functionality to the `play_agent.py` script.
    *   Installed `captum` and enhanced the `evaluate.py` script to visualize saliency maps.

12. **Configuration:**
    *   Created the `configs` directory and added `dqn_config.py` and `ddqn_config.py` for managing hyperparameters.

13. **Bug Fixing and Refinements:**
    *   Corrected model and optimizer instantiation in `train.py`.
    *   Fixed several issues in `evaluate.py` including redundant model loading and incorrect rendering.
    *   Added the `JoypadSpace` wrapper and corrected the human takeover logic in `play_agent.py`.
    *   Corrected the `dueling_agent.py` factory function to properly use the model and target_model arguments.
    *   Updated `reward_shaping.py` to include rewards for getting power-ups.
    *   Updated `base_agent.py` to handle Mario's size and firing ability.
    *   Fixed a `KeyError` in `train.py` by initializing the `info` dictionary.
    *   Fixed a `TypeError` in `train.py` by passing the `info` dictionary to the `select_action` method.

14. **Recording and Logging Enhancements:**
    *   Modified `train.py` to change the recording frequency from every 50 episodes to every 10.
    *   Updated `visualizer.py` to overlay episode metrics (episode number, total reward) onto recordings.
    *   Created `test_recording.py` to allow for immediate testing of the recording functionality.
    *   Fixed a `TypeError` in `visualizer.py` by passing the `info` dictionary to the `select_action` method.
    *   Fixed a `ZeroDivisionError` in `test_recording.py` by setting `epsilon_decay` to a non-zero value.
    *   Fixed a `RuntimeError` in `base_agent.py` by ensuring the state tensor is correctly normalized.
    *   Adjusted the text overlay position in `visualizer.py` to prevent it from obscuring important information.
    *   Implemented an agent labeling system in `train.py` and `test_recording.py` to organize recordings, models, and TensorBoard logs.
    *   Enhanced `behavior_tracker.py` and `train.py` to display more detailed training information in the console, including the current stage and both per-episode and all-time max x-positions.

15. **Advanced Training Workflow:**
    *   **Decoupled Training and Hyperparameter Search:** To create a more organized and repeatable workflow, we separated the process of finding optimal hyperparameters from the main agent training. This was achieved by introducing a `--find-hyperparameters` flag that puts the script into a dedicated search mode.
    *   **Flexible Command-Line Interface:** We implemented a more intuitive set of command-line arguments to give the user full control over the training process. This includes the ability to specify the number of generations and steps for the GA, load pre-trained models, and select a hyperparameter file for training.
    *   **Persistent Hyperparameters:** To avoid losing the results of a long hyperparameter search, we enhanced the GA to save the best-found hyperparameters to a timestamped JSON file after each generation. This provides a safeguard against interruptions and allows us to track the evolution of the hyperparameters over time. We initially considered a more complex naming convention to handle potential filename collisions, but opted for a simpler timestamp-based approach to keep the implementation clean and straightforward.
    *   **Automated Workflow:** We added a feature to automatically save the final model at the end of a training run and then immediately launch the evaluation script. This streamlines the process of training and evaluating an agent, making it easier to iterate on new ideas.

16. **Reward Shaping and Agent Behavior:**
    *   **Iterative Refinement:** We engaged in a detailed, iterative process of refining the `RewardShaper` to address specific, observable behaviors in the agent. This involved a collaborative dialogue where we analyzed the agent's performance and jointly decided on the best way to adjust the reward function.
    *   **Addressing Stagnation and Inefficiency:** We identified that the agent was often hesitant to move forward and would sometimes get stuck in loops. To address this, we implemented a series of increasingly aggressive reward structures:
        *   We significantly increased the reward for moving right to make forward progress the most attractive option.
        *   We introduced a penalty for moving left to explicitly discourage backtracking.
        *   We implemented a penalty for idling to prevent the agent from getting stuck in one place.
        *   We added a "stuck" detector that provides a large, targeted reward for jumping when the agent is unable to move forward, teaching it to overcome obstacles.
    *   **Amplifying the Reward Signal:** We discussed the impact of proportionally scaling all rewards and punishments. We concluded that while it doesn't change the relative value of actions, it amplifies the "volume" of the feedback, making the consequences of the agent's actions more significant and encouraging it to learn more quickly. Based on this, we dramatically increased the magnitude of all rewards and penalties to create a very steep "reward gradient."
    *   **Dynamic Visual Feedback:** To get a better sense of how the agent was learning, we adjusted the recording frequency to provide more frequent visual feedback on its performance. This allowed us to more quickly identify and address issues with the agent's behavior.

17. **Advanced Reward Engineering and Training Workflow (Second Iteration):**
    *   **Problem: Agent Exploiting Rightward Movement:** We observed that the agent could accumulate rewards by simply moving back and forth, as any rightward movement was rewarded, regardless of whether it made actual progress.
        *   **Solution:** We introduced the concept of a `max_x_pos` for each episode. The agent is now only rewarded for exceeding this value, incentivizing genuine forward progress. The penalty for moving left was initially kept to discourage backtracking.

    *   **Problem: Agent Gets Stuck on Obstacles:** A major issue was the agent's inability to perform the high jumps necessary to clear pipes. It would often perform small, ineffective hops.
        *   **Solution: The `ProlongedJumpWrapper`:** We designed a new environment wrapper to transform any jump action selected by the agent into a full-height jump. This was achieved by forcing the jump action to be repeated for 20 consecutive frames.
        *   **Debugging the Wrapper:** The initial implementation of this wrapper caused a `ValueError: cannot step in a done environment!` because it conflicted with the `SkipFrame` wrapper. Both wrappers had internal loops that were not synchronized. After several attempts to fix the timing, we re-architected the `ProlongedJumpWrapper` to be stateful. Instead of looping internally, it now intercepts a jump action and forces that action to be repeated for the subsequent steps, a much cleaner solution that resolved the conflict.

    *   **Problem: Inefficient Training and Evaluation Workflow:**
        *   **Issue 1: Epsilon Decay:** We identified that running shorter training sessions with the `--total_steps` argument did not automatically adjust the `epsilon_decay` rate. This resulted in the agent being far too exploratory at the end of shorter runs.
        *   **Solution 1:** We made the training script "smarter." It now detects if `--total_steps` is manually set and automatically scales `epsilon_decay` and `eval_interval` to match, ensuring a consistent learning trajectory regardless of the run's length.
        *   **Issue 2: Unreliable Evaluation:** The `evaluate` function only ran for a single, purely greedy episode, making the results highly susceptible to luck. It also caused a crash by creating a new, unwrapped environment with the wrong observation shape.
        *   **Solution 2:** We completely overhauled the `evaluate` function. It now runs for multiple episodes (e.g., 10) and uses an epsilon-greedy policy (with ε=0.05) for more realistic and stable results. It also correctly uses the existing, fully-wrapped environment passed from the training loop, fixing the crash.

    *   **Problem: Visualizer Causing Crashes and Bugs:**
        *   **Issue 1:** The `Visualizer` was resetting the main training environment, desynchronizing the state and causing the same `ValueError` as the wrapper conflict.
        *   **Issue 2:** The visualization logic was being called on every step of an episode, resulting in hundreds of duplicate recordings.
        *   **Solution:** We refactored the `Visualizer` to create its own, completely isolated environment instance for recording. This prevents any interference with the training loop. We also moved the call to the `Visualizer` inside the `if done:` block in the training loop, ensuring that recordings are generated only once at the end of a completed episode.

    *   **Iterative Reward Tuning (A Summary of Scrapped Ideas):**
        *   We initially **dramatically increased the reward for `max_x_pos` progress** (by 20x). We observed this made the agent *too* greedy for forward progress, causing it to ignore other important tasks like killing enemies. We reverted this change.
        *   We then **heavily increased the penalty for moving left**. This also proved to be too restrictive, preventing the agent from learning to maneuver around obstacles. We reduced this penalty to be symmetrical with the forward-progress reward.
        *   To teach the agent to back up for a running start, we **added a large reward for moving left *only when stuck***. This gives the agent a viable, secondary strategy to overcome obstacles without punishing all leftward movement.
        *   Finally, to encourage more interaction with the environment, we **significantly increased the reward multiplier for gaining score** and the **penalties for dying/timing out**, making these events much more impactful on the agent's decision-making.
