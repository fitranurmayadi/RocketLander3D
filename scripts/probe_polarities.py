import gymnasium as gym
import rocket_lander
import pybullet as p
import numpy as np
import time

def probe_physics():
    env = gym.make("RocketLander-v0", render_mode=None)
    obs, _ = env.reset()
    
    print("\n--- PROBING GIMBAL PITCH (+1.0) ---")
    action = np.zeros(16)
    action[0] = 0.5 # Some throttle
    action[1] = 1.0 # Max Gimbal P
    action[3:16] = -1.0 # RCS OFF
    
    for _ in range(10):
        obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"Vel X: {obs[7]:.4f}, Vel Y: {obs[8]:.4f}")
    pitch = p.getEulerFromQuaternion(obs[3:7])[1]
    print(f"Final Pitch (rad): {pitch:.4f}")
    
    obs, _ = env.reset()
    print("\n--- PROBING GIMBAL ROLL (+1.0) ---")
    action = np.zeros(16)
    action[0] = 0.5
    action[2] = 1.0 # Max Gimbal R
    action[3:16] = -1.0
    
    for _ in range(10):
        obs, reward, terminated, truncated, info = env.step(action)
        
    print(f"Vel X: {obs[7]:.4f}, Vel Y: {obs[8]:.4f}")
    roll = p.getEulerFromQuaternion(obs[3:7])[0]
    print(f"Final Roll (rad): {roll:.4f}")

    env.close()

if __name__ == "__main__":
    probe_physics()
