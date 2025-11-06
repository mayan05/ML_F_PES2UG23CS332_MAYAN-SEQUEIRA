from agent import DQNAgent
from environment import MarioEnvironment
import time


def play_episode(agent, env, render=True, max_steps=10000):
    state, info = env.reset()
    total_reward = 0
    steps = 0
    
    print("Playing episode...")
    print(f"Starting position: x={info.get('x_pos', 'N/A')}")
    
    for step in range(max_steps):
        action = agent.act(state, training=False)
        next_state, reward, done, info = env.step(action)
        
        state = next_state
        total_reward += reward
        steps += 1
        
        if render:
            env.render()
            time.sleep(0.01)
        
        if step % 100 == 0:
            print(f"Step {step}, Reward: {total_reward:.2f}, X Position: {info.get('x_pos', 'N/A')}")
        
        if done:
            print(f"Episode finished at step {step}")
            print(f"Final reward: {total_reward:.2f}")
            print(f"Final X position: {info.get('x_pos', 'N/A')}")
            break
    
    return total_reward, steps


def demo(model_path='models/best_model.pth'):
    env = MarioEnvironment(render_mode='human')
    agent = DQNAgent(env.state_shape, env.n_actions)
    
    try:
        agent.load(model_path)
        print(f"Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Model file {model_path} not found. Please train the agent first.")
        return
    
    print("Starting demo...")
    print("The agent will play Mario. Watch the game window!")
    
    for episode in range(3):
        print(f"\n--- Episode {episode + 1} ---")
        reward, steps = play_episode(agent, env, render=True)
        print(f"Episode {episode + 1} completed: Reward = {reward:.2f}, Steps = {steps}")
        time.sleep(2)
    
    env.close()
    print("Demo complete!")


if __name__ == "__main__":
    demo()

