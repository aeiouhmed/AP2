# Testing Plan

This plan outlines the steps to test the functionality of the Genetic-Optimized Deep Reinforcement Learning AI for Super Mario Bros project.

## 1. Human Playable Test

1.  **Objective:** Verify that the environment is still working correctly with human control.
2.  **Command:** `python -m mario_rl.play_human`
3.  **Expected Outcome:** The game should launch, and you should be able to control Mario using the WASD keys.

## 2. Training

1.  **Objective:** Verify that the training process starts without errors, logs are created, and recordings are saved correctly.
2.  **Modification:** I will first modify the `train.py` script to render the first 50 episodes with overlays.
3.  **Command:** `python -m mario_rl.train --agent dqn --dueling --agent_name test_agent`
4.  **Expected Outcome:**
    *   The training process should start without any errors.
    *   A new folder named `test_agent` should be created in the `recordings` directory.
    *   The first 50 episodes should be rendered with overlays, and the recordings should be saved in the `recordings/test_agent` folder.
    *   After the 50th episode, the rendering should stop, and the training should continue with only the metrics being printed to the console.
    *   A new log file should be created in the `runs` directory.

## 3. Logging and Visualization

1.  **Objective:** Verify that all metrics are being logged correctly and that the recordings are being generated.
2.  **Command:** `tensorboard --logdir runs`
3.  **Expected Outcome:**
    *   TensorBoard should launch, and you should be able to see the training logs.
    *   All metrics, including the rolling statistics, should be logged correctly.
    *   The recordings in the `recordings/test_agent` folder should be playable and show the agent's gameplay with the correct overlays.

## 4. Evaluation

1.  **Objective:** Verify that the evaluation script can evaluate a trained model and generate the correct output.
2.  **Command:** `python -m mario_rl.evaluate --model_paths models/dqn_dueling_100000.pth --saliency`
3.  **Expected Outcome:**
    *   The evaluation script should run without errors.
    *   The `evaluation_results.csv` file should be created and contain the total reward for each level.
    *   The saliency maps should be displayed for each step of the evaluation.

## 5. Playback

1.  **Objective:** Verify that the playback script can load a trained model and that the human takeover functionality works.
2.  **Command:** `python -m mario_rl.play_agent --model_path models/dqn_dueling_100000.pth --human_takeover`
3.  **Expected Outcome:**
    *   The agent should play the game.
    *   Pressing the 'h' key should allow you to take over control of the agent.
