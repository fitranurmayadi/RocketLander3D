"""
Learning Lesson 1: Gymnasium & PyBullet Basics

This script introduces the Gymnasium API, the observation/action space, 
and how to run a basic physics simulation loop.

Concepts covered:
1. Environment creation and resetting.
2. Understanding the observation vector (state variables).
3. Understanding the action vector (control inputs).
4. Running a step loop and capturing keyboard events for interactive control.
"""

import gymnasium as gym
import pybullet as p
import numpy as np
import time

# 1. Import our custom rocket_lander package to register the Gymnasium environment
try:
    import rocket_lander
except ImportError:
    print("Please make sure RocketLander is installed or sys.path is configured correctly.")

def main():
    print("==================================================")
    print("Lesson 1: Gymnasium & PyBullet Basics")
    print("==================================================")
    
    # 2. Instantiate the environment
    # We use v0 here (randomized position spawn, standard environment)
    env = gym.make("RocketLander-v0", render_mode="human")
    
    # Let's inspect the action and observation space
    print(f"Action Space:      {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    
    # 3. Reset the environment to begin simulation
    # env.reset() returns: (initial_observation, info_dictionary)
    obs, info = env.reset()
    
    print("\n--- Observation Mapping ---")
    print(f"Position (X, Y, Z):      {obs[0:3]}")
    print(f"Orientation (Quat):      {obs[3:7]}")
    print(f"Velocity (Vx, Vy, Vz):   {obs[7:10]}")
    print(f"Ang Vel (Wx, Wy, Wz):    {obs[10:13]}")
    print(f"Ground Contacts (4 legs):{obs[13:17]}")
    print(f"Throttle State:          {obs[17]}")
    print(f"Fuel Remaining:          {obs[18]}")
    
    print("\n--- Interactive Keyboard Mapping ---")
    print("Click on the PyBullet simulation window, then press:")
    print("  [W] / [S] : Increase / Decrease Main Engine Throttle")
    print("  [A] / [D] : Roll left / right (RCS)")
    print("  [UP] / [DOWN] : Pitch down / up (RCS)")
    print("  [Q] / [E] : Yaw left / right (RCS)")
    print("  [Space]   : Deploy Landing Legs")
    print("==================================================")
    
    # Initialize interactive control variables
    throttle = 0.0
    u_roll = 0.0
    u_pitch = 0.0
    u_yaw = 0.0
    legs_deploy = -1.0 # -1.0 is stowed, 1.0 is deployed
    
    dt = 1.0 / 60.0
    
    try:
        while True:
            # 4. Get keyboard events from PyBullet GUI
            keys = p.getKeyboardEvents()
            
            # W / S: Throttle
            if ord('w') in keys and keys[ord('w')] & p.KEY_IS_DOWN:
                throttle = min(1.0, throttle + 0.01)
            elif ord('s') in keys and keys[ord('s')] & p.KEY_IS_DOWN:
                throttle = max(0.0, throttle - 0.01)
                
            # A / D: Roll
            if ord('a') in keys and keys[ord('a')] & p.KEY_IS_DOWN:
                u_roll = -0.8
            elif ord('d') in keys and keys[ord('d')] & p.KEY_IS_DOWN:
                u_roll = 0.8
            else:
                u_roll = 0.0
                
            # UP / DOWN: Pitch
            if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
                u_pitch = 0.8
            elif p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
                u_pitch = -0.8
            else:
                u_pitch = 0.0
                
            # Q / E: Yaw
            if ord('q') in keys and keys[ord('q')] & p.KEY_IS_DOWN:
                u_yaw = -0.8
            elif ord('e') in keys and keys[ord('e')] & p.KEY_IS_DOWN:
                u_yaw = 0.8
            else:
                u_yaw = 0.0
                
            # Space: Deploy legs
            if ord(' ') in keys and keys[ord(' ')] & p.KEY_WAS_TRIGGERED:
                legs_deploy = 1.0 if legs_deploy < 0.0 else -1.0
                print(f"Legs state changed to: {'DEPLOYED' if legs_deploy > 0 else 'STOWED'}")
                
            # 5. Build action vector
            # The environment expects action shape (16,) mapped as:
            # action[0] = Throttle command (-1 to 1 maps to 0 to 1)
            # action[1:3] = Gimbal Roll & Pitch (we disable this by keeping it at 0.0)
            # action[3:15] = RCS thrusters. We can simplify this by setting RCS commands.
            # Fortunately, the environment also accepts a mixer-style or direct rcs mapping.
            # Let's map it exactly as our mixer does:
            # - action[0]: throttle (mapped to [-1.0, 1.0])
            # - action[1, 2]: gimbal pitch, roll (set to 0.0)
            # - action[3:15]: individual RCS thruster commands (either 0.0 or 1.0)
            # To simplify, the env mixer handles roll, pitch, yaw inputs:
            action = np.zeros(16)
            
            # Map throttle to action[0] range [-1.0, 1.0]
            action[0] = throttle * 2.0 - 1.0
            
            # Simple Mixer Logic for RCS:
            # action[3..14] correspond to the 12 RCS thrusters:
            # - Roll (+): thrusters 3, 6 active; Roll (-): thrusters 4, 5 active
            # - Pitch (+): thrusters 7, 10 active; Pitch (-): thrusters 8, 9 active
            # - Yaw (+): thrusters 11, 14 active; Yaw (-): thrusters 12, 13 active
            if u_roll > 0.1:
                action[3] = action[6] = 1.0
            elif u_roll < -0.1:
                action[4] = action[5] = 1.0
                
            if u_pitch > 0.1:
                action[7] = action[10] = 1.0
            elif u_pitch < -0.1:
                action[8] = action[9] = 1.0
                
            if u_yaw > 0.1:
                action[11] = action[14] = 1.0
            elif u_yaw < -0.1:
                action[12] = action[13] = 1.0
                
            # Landing legs state: action[15] (legs_deploy >= 0.0 triggers deployment)
            action[15] = legs_deploy
            
            # 6. Apply step to the environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Print current telemetry in terminal
            print(f"\rAlt: {obs[2]:.2f}m | Vx: {obs[7]:.2f}m/s | Vy: {obs[8]:.2f}m/s | Vz: {obs[9]:.2f}m/s | Fuel: {obs[18]*100:.1f}% | Throttle: {throttle*100:.0f}%", end="")
            
            if terminated or truncated:
                print("\nEnvironment terminated! Resetting...")
                obs, info = env.reset()
                throttle = 0.0
                legs_deploy = -1.0
                
            # Keep the simulation running at 60Hz
            time.sleep(dt)
            
    except KeyboardInterrupt:
        print("\nSimulation ended by user.")
    finally:
        env.close()

if __name__ == "__main__":
    main()
