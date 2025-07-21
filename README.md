# Project Command-Line Interface (CLI) Guide

This guide provides the commands and instructions for running the different components of the Mario RL project.

---

### 1. Run the Agent Training Process

This command starts the main training loop for a reinforcement learning agent.

**Usage:**
```bash
python -m mario_rl.train [OPTIONS]
```

**Key Options:**
*   `--agent_label <NAME>`: **(Required)** A unique name for your agent (e.g., `my_mario_agent`). This is used for saving models, logs, and recordings.
*   `--total_steps <NUMBER>`: The total number of steps to train the agent for. **Note:** This will also automatically scale `epsilon_decay` and `eval_interval` to match.
*   `--load_model <PATH>`: Path to a pre-trained model file (e.g., `models/my_mario_agent_final.pth`) to continue training from.
*   `--dueling`: Use a Dueling DQN architecture. Must be used if loading a dueling model.
*   `--rec-frequency <NUMBER>`: How often to save a gameplay recording (e.g., `10` saves a recording every 10 episodes).
*   `--render`: Add this flag to watch the agent train in real-time.

**Example:**
```bash
# Train a new dueling agent named 'SuperLearner' for 50,000 steps
python -m mario_rl.train --agent_label SuperLearner --total_steps 50000 --dueling

# Continue training 'SuperLearner' for another 50,000 steps, saving a video every 5 episodes
python -m mario_rl.train --agent_label SuperLearner_v2 --load_model models/SuperLearner_final.pth --total_steps 50000 --dueling --rec-frequency 5
```

---

### 2. Run the Hyperparameter Search Process

This command uses a genetic algorithm to find optimal hyperparameters for your agent.

**Usage:**
```bash
python -m mario_rl.train --find_hyperparameters [OPTIONS]
```

**Key Options:**
*   `--ga_generations <NUMBER>`: The number of generations the genetic algorithm should run for (default: 5).
*   `--ga_steps <NUMBER>`: The number of training steps each agent in the population is evaluated for (default: 10,000).

**Example:**
```bash
# Run a hyperparameter search for 10 generations, with each evaluation lasting 20,000 steps
python -m mario_rl.train --find_hyperparameters --ga_generations 10 --ga_steps 20000
```

---

### 3. Play the Game as a Human

This script allows you to play the game directly using your keyboard to test the environment.

**Usage:**
```bash
python -m mario_rl.play_human
```

**Controls:**
*   **W:** Jump
*   **A:** Move Left
*   **D:** Move Right
*   **Q:** Quit

---

### 4. Watch a Trained Agent Play

This script loads a saved agent and runs it in the environment for you to watch.

**Usage:**
```bash
python -m mario_rl.play_agent --model_path <PATH> [OPTIONS]
```

**Key Options:**
*   `--model_path <PATH>`: **(Required)** The full path to the saved model file (e.g., `models/SuperLearner_final.pth`).
*   `--dueling`: **(Required if applicable)** You must include this flag if the saved model was trained with a Dueling DQN architecture.
*   `--level <WORLD-STAGE>`: Specify a particular level to play (e.g., `SuperMarioBros-2-1-v0`). If not provided, it will attempt to play through all levels.
*   `--human_takeover`: Allows you to press 'H' during gameplay to take control of the agent with the keyboard.

**Example:**
```bash
# Watch the 'SuperLearner' agent play World 4-2
python -m mario_rl.play_agent --model_path models/SuperLearner_final.pth --dueling --level SuperMarioBros-4-2-v0
```

---

### 5. Activate TensorBoard

This command starts the TensorBoard server, which provides a web-based interface to visualize training metrics.

**Usage:**
```bash
tensorboard --logdir runs
```

**Instructions:**
1.  Run this command from your project's root directory (`c:/Users/user/Desktop/AP2`).
2.  Open your web browser and navigate to the URL provided in the terminal (usually `http://localhost:6006/`).
3.  You will see interactive charts for training loss, episode rewards, and evaluation rewards for all of your agents.
