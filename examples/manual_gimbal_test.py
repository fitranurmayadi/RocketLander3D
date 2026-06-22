import gymnasium as gym
import rocket_lander
import pybullet as p
import numpy as np
import math

def manual_test():
    env = gym.make("RocketLander-v0", render_mode=None)
    
    # 1. Pitch Test
    print("\n--- PITCH TEST (Raw Action[1] = 0.5) ---")
    obs, _ = env.reset()
    r_initial = p.getEulerFromQuaternion(obs[3:7])[1]
    
    action = np.full(16, -1.0)
    action[0] = 1.0 # Max thrust
    action[1] = 0.5 # Pos Pitch Gimbal
    
    for _ in range(10):
        obs, _, _, _, _ = env.step(action)
    
    r_final = p.getEulerFromQuaternion(obs[3:7])[1]
    diff = math.degrees(r_final - r_initial)
    print(f"Initial Pitch: {math.degrees(r_initial):.2f}")
    print(f"Final Pitch (10 steps): {math.degrees(r_final):.2f}")
    print(f"Delta: {diff:.2f}")
    if diff > 0:
        print("RESULT: Positive Action -> Positive Pitch Change (Nose UP?)")
    else:
        print("RESULT: Positive Action -> Negative Pitch Change (Nose DOWN?)")

    # 2. Roll Test
    print("\n--- ROLL TEST (Raw Action[2] = 0.5) ---")
    obs, _ = env.reset()
    r_initial = p.getEulerFromQuaternion(obs[3:7])[0]
    
    action = np.full(16, -1.0)
    action[0] = 1.0 # Max thrust
    action[2] = 0.5 # Pos Roll Gimbal
    
    for _ in range(10):
        obs, _, _, _, _ = env.step(action)
    
    r_final = p.getEulerFromQuaternion(obs[3:7])[0]
    diff = math.degrees(r_final - r_initial)
    print(f"Initial Roll: {math.degrees(r_initial):.2f}")
    print(f"Final Roll (10 steps): {math.degrees(r_final):.2f}")
    print(f"Delta: {diff:.2f}")
    if diff > 0:
        print("RESULT: Positive Action -> Positive Roll Change (Roll LEFT?)")
    else:
        print("RESULT: Positive Action -> Negative Roll Change (Roll RIGHT?)")

    env.close()

if __name__ == "__main__":
    manual_test()
