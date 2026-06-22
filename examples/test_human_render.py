
import gymnasium as gym
import numpy as np
import time
import os
import sys

# Force local import
curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(curr_dir)
sys.path.insert(0, root_dir)

import rocket_lander

def test_human_render():
    print("DEBUG: Initializing Env in human mode...")
    env = gym.make("RocketLander-v0", render_mode="human")
    obs, _ = env.reset()
    
    print("DEBUG: Simulation running for 500 steps. Take control or watch the rocket.")
    for i in range(50000):
        # Random action but with some thrust to see flames
        action = env.action_space.sample()
        action[0] = 0.5 # Constant half thrust
        
        obs, reward, terminated, truncated, _ = env.step(action)
        
        env.render()
        
        if terminated or truncated:
            obs, _ = env.reset()
            
    env.close()

if __name__ == "__main__":
    test_human_render()
