from nes_py.wrappers import JoypadSpace
import gymnasium
from gymnasium.wrappers import FrameStack
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import numpy as np
import cv2
from collections import deque


class MarioEnvironment:
    def __init__(self, env_name='SuperMarioBros-v0', frame_skip=4, stack_frames=4, render_mode=None):
        # Create environment with optional render_mode
        if render_mode:
            self.env = gym_super_mario_bros.make(env_name, render_mode=render_mode, apply_api_compatibility=True)
        else:
            self.env = gym_super_mario_bros.make(env_name, apply_api_compatibility=True)
        
        self.env = JoypadSpace(self.env, SIMPLE_MOVEMENT)
        self.frame_skip = frame_skip
        self.stack_frames = stack_frames
        self.frame_stack = deque(maxlen=stack_frames)
        self.n_actions = self.env.action_space.n
        self.state_shape = (stack_frames, 84, 84)
    
    def preprocess_frame(self, frame):
        """
        Preprocess the frame: convert to grayscale, resize to 84x84, normalize to [0,1]
        """
        # Convert RGB to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Resize to 84x84
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1]
        normalized = resized / 255.0
        
        return normalized
    
    def reset(self):
        """
        Reset the environment and initialize frame stack
        """
        # Reset the environment
        result = self.env.reset()
        
        # Handle both old and new gym API
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}
        
        # Preprocess the initial frame
        processed_frame = self.preprocess_frame(obs)
        
        # Fill the frame stack with the initial frame (repeated)
        self.frame_stack.clear()
        for _ in range(self.stack_frames):
            self.frame_stack.append(processed_frame)
        
        # Stack frames and return
        state = np.array(self.frame_stack)
        
        return state, info
    
    def step(self, action):
        """
        Take action and return stacked frames
        """
        total_reward = 0
        done = False
        info = {}
        
        # Execute action for frame_skip steps and accumulate rewards
        for _ in range(self.frame_skip):
            result = self.env.step(action)
            
            # Handle both old and new gym API
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:
                obs, reward, done, info = result
            
            total_reward += reward
            
            if done:
                break
        
        # Preprocess the new frame
        processed_frame = self.preprocess_frame(obs)
        
        # Add to frame stack
        self.frame_stack.append(processed_frame)
        
        # Stack frames
        next_state = np.array(self.frame_stack)
        
        return next_state, total_reward, done, info
    
    def render(self, mode='human'):
        return self.env.render()
    
    def close(self):
        self.env.close()