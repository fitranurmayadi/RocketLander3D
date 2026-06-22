
import gymnasium as gym
import time
import numpy as np
import pybullet as p
import rocket_lander

def debug_ascent():
    # Force headless mode to print to stdout cleanly
    env = gym.make("RocketLander-v0", render_mode=None)
    
    # Spawn high
    options = {
        "initial_pos": [-1000.0, -1000.0, 1000.0],
        "initial_orn": p.getQuaternionFromEuler([0.785, 0.785, 0.785])
    }
    obs, _ = env.reset(options=options)
    
    print("Step | Z | Vz | Action | Throttle | Force | G")
    
    # Action: -1.0 throttle (should be 0 force)
    action = np.zeros(7, dtype=np.float32)
    action[0] = -1.0 
    
    for i in range(10):
        # Peak at internal state if possible
        throttle = (action[0] + 1.0) / 2.0
        force_mag = throttle * env.unwrapped.MAIN_ENGINE_POWER
        
        obs, reward, terminated, truncated, info = env.step(action)
        z = obs[2]
        vz = obs[9]
        
        print(f"{i:4d} | {z:6.2f} | {vz:6.2f} | {action[0]:6.2f} | {throttle:6.2f} | {force_mag:6.2f}")
        
    env.close()

if __name__ == "__main__":
    debug_ascent()
