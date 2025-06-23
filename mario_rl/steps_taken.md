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
