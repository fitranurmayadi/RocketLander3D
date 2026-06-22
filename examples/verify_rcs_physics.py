import gymnasium as gym
import rocket_lander
import pybullet as p
import numpy as np
import time

def test_rcs():
    env = gym.make("RocketLander-v0", render_mode=None)
    env.reset()
    
    print("\n--- RCS TORQUE TEST ---")
    # Pairs based on pid_controller_basic.py logic
    tests = [
        ("PITCH DOWN (0,5)", [0, 5]),
        ("PITCH UP (1,4)", [1, 4]),
        ("ROLL LEFT (2,7)", [2, 7]),
        ("ROLL RIGHT (3,6)", [3, 6]),
        ("YAW RIGHT (8,10)", [8, 10]),
        ("YAW LEFT (9,11)", [9, 11]),
    ]
    
    for name, rcs_indices in tests:
        obs, _ = env.reset()
        # Zero gravity for pure observation
        p.setGravity(0, 0, 0)
        
        # Initial state
        quat = obs[3:7]
        rpy_init = p.getEulerFromQuaternion(quat)
        
        # Fire RCS
        action = np.zeros(16)
        action[3:16] = -1.0 # Default OFF
        for idx in rcs_indices:
            action[3 + idx] = 1.0
            
        for _ in range(10):
            obs, _, _, _, _ = env.step(action)
            
        quat_new = obs[3:7]
        rpy_new = p.getEulerFromQuaternion(quat_new)
        
        diff = [np.degrees(rpy_new[i] - rpy_init[i]) for i in range(3)]
        print(f"{name}: Roll={diff[0]:.2f}, Pitch={diff[1]:.2f}, Yaw={diff[2]:.2f}")
        
    env.close()

if __name__ == "__main__":
    test_rcs()
