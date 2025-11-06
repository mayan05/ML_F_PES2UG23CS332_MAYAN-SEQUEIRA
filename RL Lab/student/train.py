import numpy as np
from agent import DQNAgent
from environment import MarioEnvironment
import matplotlib.pyplot as plt
import os


def train_agent(episodes=1000, max_steps=10000, update_frequency=4, target_update_frequency=1000):
    env = MarioEnvironment()
    agent = DQNAgent(env.state_shape, env.n_actions)
    
    scores = []
    episode_rewards = []
    best_score = -float('inf')
    
    print("Starting training...")
    print(f"State shape: {env.state_shape}")
    print(f"Number of actions: {env.n_actions}")
    
    for episode in range(episodes):
        state, info = env.reset()
        total_reward = 0
        steps = 0
        
        for step in range(max_steps):
            action = agent.act(state, training=True)
            next_state, reward, done, info = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if len(agent.memory) > 32 and step % update_frequency == 0:
                agent.replay(batch_size=32)
            
            if done:
                break
        
        if episode % target_update_frequency == 0 and episode > 0:
            agent.update_target_network()
        
        episode_rewards.append(total_reward)
        scores.append(steps)
        
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_score = np.mean(scores[-10:])
            print(f"Episode {episode}, Average Reward: {avg_reward:.2f}, Average Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.3f}")
        
        if total_reward > best_score:
            best_score = total_reward
            os.makedirs('models', exist_ok=True)
            agent.save('models/best_model.pth')
    
    env.close()
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(1, 2, 2)
    plt.plot(scores)
    plt.title('Episode Length (Steps)')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    print("Training complete. Results saved to training_results.png")
    
    return agent


if __name__ == "__main__":
    agent = train_agent(episodes=500, max_steps=5000)

