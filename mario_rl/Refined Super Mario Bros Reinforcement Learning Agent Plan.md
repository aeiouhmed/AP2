# Refined Super Mario Bros Reinforcement Learning Agent Plan

Based on your feedback, I'll outline a comprehensive implementation plan that includes both DQN and DDQN with Dueling architecture, extensive logging, and visualization of agent progress.

## Library Choice: PyTorch

I recommend using PyTorch for this project for several reasons:

- __Dynamic computation graph__: Easier to debug and understand what's happening during training
- __Pythonic design__: More intuitive for complex architectures like Dueling DQN
- __Research-friendly__: Easier to implement and experiment with custom components
- __Good visualization integration__: Works well with libraries like Tensorboard for tracking metrics

## Project Structure

```javascript
mario_rl/
├── agents/
│   ├── base_agent.py         # Abstract base class for agents
│   ├── dqn_agent.py          # DQN implementation
│   ├── ddqn_agent.py         # Double DQN implementation
│   └── dueling_agent.py      # Dueling architecture implementation
├── environment/
│   ├── wrappers.py           # Custom environment wrappers
│   └── reward_shaping.py     # Custom reward functions
├── models/
│   ├── networks.py           # Neural network architectures
│   └── replay_buffer.py      # Experience replay implementation
├── utils/
│   ├── logger.py             # Logging utilities
│   ├── visualizer.py         # Visualization tools
│   └── behavior_tracker.py   # Track agent learning milestones
├── configs/
│   ├── dqn_config.py         # DQN hyperparameters
│   └── ddqn_config.py        # DDQN hyperparameters
├── train.py                  # Main training script
├── play_human.py             # Human playable test
├── evaluate.py               # Evaluation script
└── play_agent.py             # Run trained agent
```

## Phase 1: Environment Setup and Testing

### Human Playable Test (`play_human.py`)

- Map keyboard inputs to game actions
- Display controls on screen
- Allow for recording gameplay for potential imitation learning

### Environment Wrappers (`environment/wrappers.py`)

1. __PreprocessFrame__: Convert to grayscale, resize (84x84)

2. __FrameStack__: Stack 4 frames to capture motion

3. __SkipFrame__: Skip frames for computational efficiency

4. __RewardShaper__: Implement custom reward function:

   - Positive reward proportional to rightward velocity
   - Large reward for level completion (e.g., +500)
   - Small negative reward per timestep (-0.1) to encourage speed
   - Larger negative reward for death (-100)
   - Bonus rewards for coins, enemies defeated, etc.

## Phase 2: Agent Architecture

### Neural Network Models (`models/networks.py`)

1. __Base CNN__: Convolutional layers for processing game frames

   ```javascript
   Conv2D(4, 32, 8, stride=4) -> ReLU
   Conv2D(32, 64, 4, stride=2) -> ReLU
   Conv2D(64, 64, 3, stride=1) -> ReLU
   Flatten
   Linear(3136, 512) -> ReLU
   ```

2. __Standard DQN__: Base CNN + output layer

   ```javascript
   Linear(512, num_actions)
   ```

3. __Dueling DQN__: Base CNN + advantage and value streams

   ```javascript
   Value: Linear(512, 1)
   Advantage: Linear(512, num_actions)
   Q = Value + (Advantage - mean(Advantage))
   ```

### Replay Buffer (`models/replay_buffer.py`)

- Fixed-size circular buffer for experience storage
- Batch sampling for training
- Optional: Prioritized experience replay

### Agent Implementations

1. __Base Agent__ (`agents/base_agent.py`): Common functionality

   - Epsilon-greedy action selection
   - Experience collection
   - Model saving/loading

2. __DQN Agent__ (`agents/dqn_agent.py`):

   - Standard DQN algorithm
   - Target network updates
   - Loss calculation

3. __DDQN Agent__ (`agents/ddqn_agent.py`):

   - Double DQN algorithm (decoupled action selection and evaluation)
   - Reduced overestimation bias

4. __Dueling Agent__ (`agents/dueling_agent.py`):

   - Dueling network architecture
   - Can be combined with both DQN and DDQN

## Phase 3: Training Pipeline

### Training Script (`train.py`)

- Command-line arguments for algorithm selection and hyperparameters
- Training loop with periodic evaluation
- Model checkpointing
- Visualization scheduling

### Logging System (`utils/logger.py`)

Comprehensive metrics tracking:

- __Per-episode metrics__:

  - Total reward
  - Episode length
  - Max X-position reached
  - Number of deaths
  - Coins collected
  - Enemies defeated
  - Level completion success/failure

- __Training metrics__:

  - Loss values
  - Q-value statistics (min, max, mean, std)
  - Gradient norms
  - Learning rate
  - Exploration rate (epsilon)
  - Memory usage

- __Rolling statistics__:

  - Moving average of rewards (100-episode window)
  - Success rate
  - Average episode length
  - Average X-position progress

### Behavior Tracker (`utils/behavior_tracker.py`)

- Track milestone achievements:

  - First time reaching specific X-positions
  - First time defeating enemies
  - First time completing a level
  - First time using specific game mechanics (jumping, ducking)
  - Detection of strategic behaviors (waiting for enemies, backtracking)

### Visualization (`utils/visualizer.py`)

- Render agent gameplay every X iterations (configurable)
- Overlay iteration number and current metrics
- Record videos of milestone achievements
- Generate comparison videos between different training stages

## Phase 4: Evaluation and Deployment

### Evaluation Script (`evaluate.py`)

- Run trained agent through all levels
- Generate performance statistics
- Compare different algorithms (DQN vs DDQN)
- Visualize attention/saliency maps to understand agent focus

### Agent Player (`play_agent.py`)

- Load trained model
- Play through all game levels
- Display performance metrics
- Option to take over with human control

## Hyperparameter Configuration

### DQN/DDQN Configuration (`configs/`)

- Learning rate: 0.0001 - 0.00025
- Discount factor (gamma): 0.99
- Replay buffer size: 100,000 - 1,000,000
- Batch size: 32 - 128
- Target network update frequency: 1,000 - 10,000 steps
- Exploration rate (epsilon): 1.0 → 0.1 over 1,000,000 frames
- Training frames: 10,000,000+

## Implementation Timeline

1. __Week 1__: Environment setup, wrappers, and human playable test
2. __Week 2__: Neural network models and replay buffer implementation
3. __Week 3__: DQN and DDQN agent implementations
4. __Week 4__: Dueling architecture and logging system
5. __Week 5__: Training, evaluation, and hyperparameter tuning
6. __Week 6__: Analysis, visualization, and final improvements

This plan provides a comprehensive framework for implementing and comparing DQN and DDQN agents with Dueling architecture for Super Mario Bros. The extensive logging and visualization components will help track the agent's learning progress and provide insights for hyperparameter tuning.
