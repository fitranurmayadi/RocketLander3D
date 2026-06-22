import gymnasium as gym
import numpy as np
import pybullet as p
import os
import sys

# Path fix for package import
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import rocket_lander

def test_polarity():
    env = gym.make("RocketLander-v0", render_mode="DIRECT")
    obs, info = env.reset()
    
    # Test RCS Pitch (Action 3)
    print("\n--- Testing RCS Pitch (Action 3) ---")
    action = np.zeros(7)
    action[3] = 1.0 # Positive Action
    
    obs, _, _, _, _ = env.step(action)
    # Get wy (abs[11])
    wy = obs[11]
    print(f"Action[3]=1.0 -> wy: {wy:.4f}")
    if wy > 0: print("Result: Action[3]=1.0 -> POSITIVE Torque on Y")
    else: print("Result: Action[3]=1.0 -> NEGATIVE Torque on Y")

    # Test RCS Roll (Action 5)
    print("\n--- Testing RCS Roll (Action 5) ---")
    obs, info = env.reset()
    action = np.zeros(7)
    action[5] = 1.0 # Positive Action
    obs, _, _, _, _ = env.step(action)
    wx = obs[10]
    print(f"Action[5]=1.0 -> wx: {wx:.4f}")
    if wx > 0: print("Result: Action[5]=1.0 -> POSITIVE Torque on X")
    else: print("Result: Action[5]=1.0 -> NEGATIVE Torque on X")

    # Test Translation from Pitch (Tilt top to +X)
    print("\n--- Testing Translation from Pitch ---")
    obs, info = env.reset()
    # 1. Tilt rocket top to +X (Positive Pitch)
    # Actually, I can just set the base orientation.
    p.resetBasePositionAndOrientation(env.unwrapped.rocketId, [0,0,10], p.getQuaternionFromEuler([0, 0.2, 0]))
    
    # Apply Throttle
    action = np.zeros(7)
    action[0] = 1.0 # Full Throttle
    obs, _, _, _, _ = env.step(action)
    vx = obs[7]
    print(f"Pitch=0.2, Throttle=1.0 -> vx: {vx:.4f}")
    if vx > 0: print("Result: Positive Pitch -> Acceleration towards +X")
    else: print("Result: Positive Pitch -> Acceleration towards -X")

    # Repeat for Roll (Tilt top to +Y)
    print("\n--- Testing Translation from Roll ---")
    p.resetBasePositionAndOrientation(env.unwrapped.rocketId, [0,0,10], p.getQuaternionFromEuler([0.2, 0, 0]))
    obs, _, _, _, _ = env.step(action)
    vy = obs[8]
    print(f"Roll=0.2, Throttle=1.0 -> vy: {vy:.4f}")
    if vy > 0: print("Result: Positive Roll -> Acceleration towards +Y")
    else: print("Result: Positive Roll -> Acceleration towards -Y")

    env.close()

if __name__ == "__main__":
    test_polarity()
