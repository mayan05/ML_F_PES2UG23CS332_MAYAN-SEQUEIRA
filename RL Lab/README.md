# Reinforcement Learning Lab: Playing Super Mario Bros with DQN

This lab provides a hands-on introduction to Deep Reinforcement Learning using Deep Q-Networks (DQN) to train an agent that plays Super Mario Bros. The project is structured into two folders: one for students to implement their solution, and one containing the complete reference implementation.

## Table of Contents

1. Overview
2. Prerequisites
3. Installation
4. Project Structure
5. Background Theory
6. Implementation Guide
7. Running the Solution
8. Assignment Tasks
9. Evaluation
10. Troubleshooting

## Overview

In this lab, you will implement a Deep Q-Network (DQN) agent that learns to play Super Mario Bros. DQN is a value-based reinforcement learning algorithm that combines Q-Learning with deep neural networks to handle high-dimensional state spaces like image frames from video games.

The main components you will implement include:
- Environment wrapper for preprocessing game frames
- Experience replay buffer for storing and sampling transitions
- Deep Q-Network architecture using convolutional neural networks
- DQN agent with epsilon-greedy exploration and target network updates
- Training loop with proper hyperparameter tuning

## Prerequisites

Before starting this lab, you should be familiar with:
- Python programming
- Basic neural networks and deep learning concepts
- PyTorch framework
- Reinforcement learning fundamentals (Markov Decision Processes, Q-Learning)
- Convolutional Neural Networks (CNNs)

## Installation

1. Clone or download this repository to your local machine.

2. Install the required dependencies. It is recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. The installation includes:
   - gymnasium: Modern reinforcement learning environments
   - gymnasium-super-mario-bros: Super Mario Bros environment wrapper
   - nes-py: NES emulator for running the game
   - torch: PyTorch deep learning framework
   - numpy: Numerical computing
   - matplotlib: Plotting training results
   - opencv-python: Image processing for frame preprocessing

Note: If you encounter issues installing nes-py or gymnasium-super-mario-bros, you may need to install additional system dependencies. On Ubuntu/Debian:
```bash
sudo apt-get install cmake libsdl2-dev
```

## Project Structure

```
RL_lab/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── student/                  # Your implementation folder
│   ├── agent.py             # TODO: Implement DQN agent
│   ├── environment.py       # TODO: Implement environment wrapper
│   ├── train.py             # Training script (provided)
│   └── demo.py              # Demo script (provided)
└── solution/                # Reference implementation
    ├── agent.py             # Complete DQN implementation
    ├── environment.py       # Complete environment wrapper
    ├── train.py             # Training script
    └── demo.py              # Demo script
```

## Background Theory

### Reinforcement Learning Basics

Reinforcement Learning (RL) is a framework where an agent learns to make decisions by interacting with an environment. The agent receives observations (states), takes actions, and receives rewards. The goal is to maximize cumulative reward over time.

Key components:
- State (s): Current observation of the environment
- Action (a): Decision made by the agent
- Reward (r): Feedback signal from the environment
- Policy (π): Strategy the agent uses to select actions

### Q-Learning

Q-Learning is an off-policy value-based RL algorithm that learns the optimal action-value function Q(s,a), which represents the expected cumulative reward of taking action a in state s and following the optimal policy thereafter.

The Q-value update rule:
Q(s,a) = Q(s,a) + α [r + γ max Q(s',a') - Q(s,a)]

Where:
- α (alpha) is the learning rate
- γ (gamma) is the discount factor
- r is the immediate reward
- s' is the next state

### Deep Q-Network (DQN)

DQN extends Q-Learning to handle high-dimensional state spaces by using a deep neural network to approximate the Q-function. Key innovations:

1. Experience Replay: Store transitions (s, a, r, s', done) in a replay buffer and sample random batches during training to break correlations in sequential data.

2. Target Network: Use a separate target network to compute Q-targets, updated periodically. This stabilizes training by keeping target values fixed for a while.

3. Frame Stacking: Stack multiple consecutive frames to provideSubmission checklist:
- All TODO sections completed
- Training script runs successfully
- Demo script loads model and plays
- Brief report on hyperparameter experiments (optional but recommended) temporal information and motion context.

4. Epsilon-Greedy Exploration: Balance exploration (random actions) and exploitation (greedy actions) using an epsilon parameter that decays over time.

### Network Architecture

The DQN uses a CNN architecture:
- Input: Stacked grayscale frames (4, 84, 84)
- Convolutional layers: Extract spatial features from frames
- Fully connected layers: Map features to Q-values for each action

## Implementation Guide

### Step 1: Environment Wrapper (environment.py)

The environment wrapper preprocesses raw game frames and manages frame stacking.

Tasks:
1. Implement `preprocess_frame()`: Convert RGB frame to grayscale, resize to 84x84, normalize pixel values to [0,1].
2. Implement `reset()`: Reset environment, preprocess initial frame, fill frame stack with the same frame.
3. Implement `step()`: Apply action, accumulate rewards over frame_skip steps, preprocess next frame, update frame stack.

Hints:
- Use cv2.cvtColor for RGB to grayscale conversion
- Use cv2.resize for resizing frames
- Frame stack should be a deque with maxlen=stack_frames
- Return stacked frames as numpy array

### Step 2: Replay Buffer (agent.py - ReplayBuffer class)

The replay buffer stores and samples experience transitions.

Tasks:
1. Implement `push()`: Add transition (state, action, reward, next_state, done) to buffer.
2. Implement `sample()`: Randomly sample a batch of transitions and return them as separate arrays.

Hints:
- Use deque for efficient append operations
- Use random.sample() for random sampling
- Unzip the batch to separate states, actions, rewards, next_states, dones

### Step 3: DQN Agent (agent.py - DQNAgent class)

Implement the core DQN agent with key methods.

Tasks:
1. Implement `update_target_network()`: Copy Q-network weights to target network.
2. Implement `act()`: Select action using epsilon-greedy policy. If training and random < epsilon, return random action. Otherwise, forward pass through Q-network and return argmax.
3. Implement `remember()`: Store transition in replay buffer.
4. Implement `replay()`: Sample batch from replay buffer, compute Q-targets using target network, compute loss, backpropagate, update Q-network, decay epsilon.
5. Implement `save()` and `load()`: Save/load model weights using torch.save/load.

Hints:
- Convert numpy arrays to tensors before passing to network
- Use .detach() on target network outputs to stop gradients
- Q-target = reward + gamma * max(next_q_values) * (1 - done)
- Use F.mse_loss for loss computation
- Remember to zero gradients before backward pass

### Step 4: Training

The training script (train.py) is provided. It:
- Creates environment and agent
- Runs episodes, collects experiences
- Updates agent periodically
- Saves best model
- Plots training curves

Key hyperparameters:
- episodes: Number of training episodes (start with 500)
- max_steps: Maximum steps per episode (5000)
- update_frequency: How often to update network (every 4 steps)
- target_update_frequency: How often to update target network (every 1000 episodes)

## Running the Solution

To test the reference implementation:

1. Navigate to the solution folder:
```bash
cd solution
```

2. Train the agent:
```bash
python train.py
```

Training will take several hours depending on your hardware. The script will:
- Print progress every 10 episodes
- Save the best model to models/best_model.pth
- Generate training_results.png with reward and step plots

3. Run the demo:
```bash
python demo.py
```

This loads the trained model and plays 3 episodes. The demo saves gameplay videos to `demo_output/` directory so you can review the agent's performance. Each video is named with the episode number, timestamp, and reward achieved.

## Assignment Tasks

Work in the `student/` folder. Complete the following:

1. Implement `MarioEnvironment` class methods in `environment.py`:
   - `preprocess_frame()`
   - `reset()`
   - `step()`

2. Implement `ReplayBuffer` class methods in `agent.py`:
   - `push()`
   - `sample()`

3. Implement `DQNAgent` class methods in `agent.py`:
   - `update_target_network()`
   - `act()`
   - `remember()`
   - `replay()`
   - `save()` and `load()`

4. Train your agent and achieve:
   - Average reward > 200 over last 50 episodes
   - Agent progresses beyond first level section

5. Experiment with hyperparameters:
   - Try different learning rates (0.0001, 0.001, 0.00025)
   - Adjust epsilon decay rate
   - Modify network architecture
   - Report findings



## Troubleshooting

Common issues and solutions:

1. Import errors:
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version (3.8+ required)

2. Environment errors:
   - Install SDL2: `sudo apt-get install libsdl2-dev` (Linux)
   - On Mac: `brew install sdl2`
   - On Windows: Ensure Visual C++ redistributables are installed

3. CUDA/GPU issues:
   - Code works on CPU, but GPU speeds up training
   - Check PyTorch CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`

4. Training not improving:
   - Check epsilon decay: should start high (1.0) and decay slowly
   - Ensure target network updates periodically
   - Verify loss decreases over time
   - Try training for more episodes

5. Memory issues:
   - Reduce batch size in replay()
   - Reduce replay buffer capacity
   - Reduce frame stack size

6. Model not loading:
   - Ensure model was saved successfully during training
   - Check file path in demo.py
   - Verify model directory exists

## Additional Resources

- Original DQN paper: "Playing Atari with Deep Reinforcement Learning" (Mnih et al., 2013)
- Gymnasium documentation: https://gymnasium.farama.org/
- PyTorch tutorials: https://pytorch.org/tutorials/
- Super Mario Bros environment: https://github.com/Kautenja/gym-super-mario-bros

## License

This lab is for educational purposes. Super Mario Bros is a trademark of Nintendo Co., Ltd.

