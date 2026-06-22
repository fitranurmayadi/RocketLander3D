import gymnasium as gym
import rocket_lander
import pybullet as p
import numpy as np
import time

def verify_torque():
    env = gym.make("RocketLander-v0", render_mode=None)
    env.reset()
    
    # 1. Test Pitch Gimbal
    # Apply positive gimbal pitch (+0.2 rad) and 100% thrust
    # Nozzle should tilt, creating torque.
    # We measure angular velocity change.
    
    dt = 1.0/30.0
    action = np.full(16, -1.0)
    action[0] = 1.0  # Max Throttle
    action[1] = 0.5  # Positive Gimbal Pitch (~0.17 rad)
    action[2] = 0.0  # Zero Gimbal Roll
    
    print("\n--- Testing Pitch Gimbal (Action[1] = 0.5) ---")
    obs, _, _, _, _ = env.step(action)
    # AngVel is at indices 10, 11, 12 (Local X, Y, Z)
    # Wait, my env might return global or local. 
    # Let's check getBaseVelocity directly from p
    
    for i in range(5):
        obs, _, _, _, _ = env.step(action)
        lin_v, ang_v = p.getBaseVelocity(env.unwrapped.rocketId)
        print(f"Sub-step {i}: AngVel Y: {ang_v[1]:.4f}")
    
    if ang_v[1] > 0:
        print("RESULT: Positive Gimbal Pitch -> Positive Pitch Torque (Nose UP)")
    else:
        print("RESULT: Positive Gimbal Pitch -> Negative Pitch Torque (Nose DOWN)")

    # 2. Test Roll Gimbal
    env.reset()
    action = np.full(16, -1.0)
    action[0] = 1.0
    action[1] = 0.0
    action[2] = 0.5 # Positive Gimbal Roll
    
    print("\n--- Testing Roll Gimbal (Action[2] = 0.5) ---")
    for i in range(5):
        obs, _, _, _, _ = env.step(action)
        lin_v, ang_v = p.getBaseVelocity(env.unwrapped.rocketId)
        print(f"Sub-step {i}: AngVel X: {ang_v[0]:.4f}")
        
    if ang_v[0] > 0:
        print("RESULT: Positive Gimbal Roll -> Positive Roll Torque (Roll LEFT)")
    else:
        print("RESULT: Positive Gimbal Roll -> Negative Roll Torque (Roll RIGHT)")

    env.close()

if __name__ == "__main__":
    verify_torque()
