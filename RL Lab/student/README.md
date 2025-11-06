# Student Implementation: DQN Agent for Super Mario Bros

This folder contains scaffold code for you to implement. Follow the main README.md in the project root for detailed instructions.

## Tasks

Complete the following methods:

### environment.py
- `preprocess_frame()`: Convert and resize frames
- `reset()`: Reset environment and initialize frame stack
- `step()`: Take action and update frame stack

### agent.py
- `ReplayBuffer.push()`: Store transitions
- `ReplayBuffer.sample()`: Sample random batch
- `DQNAgent.update_target_network()`: Copy weights to target
- `DQNAgent.act()`: Epsilon-greedy action selection
- `DQNAgent.remember()`: Store in replay buffer
- `DQNAgent.replay()`: Train on batch
- `DQNAgent.save()`: Save model
- `DQNAgent.load()`: Load model

## Testing

Once implemented, test your code:

```bash
python train.py
```

This will train your agent. Check that:
1. No errors occur
2. Training progresses (reward increases over time)
3. Model is saved successfully

Then test the demo:

```bash
python demo.py
```

## Getting Help

- Refer to main README.md for detailed explanations
- Check solution/ folder for reference (but try to implement yourself first)
- Review DQN theory and PyTorch documentation

