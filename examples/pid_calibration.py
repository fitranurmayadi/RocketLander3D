import gymnasium as gym
import rocket_lander
import pybullet as p
import numpy as np
import time
import math
import argparse
from pid_controller_basic import RocketPIDController, PID

def run_calibration(mode="hover", no_render=False):
    print(f"\n--- PID CALIBRATION MODE: {mode.upper()} (Render: {not no_render}) ---")
    
    render_mode = "human" if not no_render else None
    env = gym.make("RocketLander-v0", render_mode=render_mode)
    
    # 1. Setup Initial Conditions based on Mode
    initial_pos = [0, 0, 300.0]
    initial_rpy = [0, 0, 0]
    
    if mode == "pitch":
        initial_rpy = [0, math.radians(20), 0]
    elif mode == "roll":
        initial_rpy = [math.radians(20), 0, 0]
    elif mode == "yaw":
        initial_rpy = [0, 0, math.radians(45)]
    
    initial_quat = p.getQuaternionFromEuler(initial_rpy)
    
    obs, info = env.reset(options={
        "initial_pos": initial_pos,
        "initial_orn": initial_quat
    })
    
    # Stability: Start with neutral velocity
    p.resetBaseVelocity(env.unwrapped.rocketId, [0, 0, 0], [0, 0, 0])
    
    controller = RocketPIDController()
    # Force target altitude to 300 for calibration
    controller.alt_pid.target = 300.0
    
    dt = 1.0 / 30.0
    
    try:
        for i in range(1500): # Longer duration for stability check
            action = controller.get_action(obs, dt)
            
            # If in hover mode, we might want to ignore horizontal drift for now
            # but usually we want all PID to work.
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            if not no_render:
                env.render()
            
            if i % 30 == 0:
                rpy_now = p.getEulerFromQuaternion(obs[3:7])
                rpy_deg = [math.degrees(a) for a in rpy_now]
                fuel_pct = obs[18] * 100.0
                vx, vy = obs[7], obs[8]
                px, py = obs[0], obs[1]
                print(f"T={i/30:.1f}s | Alt: {obs[17]:.1f}m | Vz: {obs[9]:.2f}m/s | RPY: {rpy_deg[0]:.1f}, {rpy_deg[1]:.1f}, {rpy_deg[2]:.1f}")
                print(f"  [TRANSLATION] Pos: ({px:.1f}, {py:.1f}) | Vel: ({vx:.2f}, {vy:.2f}) | Fuel: {fuel_pct:.1f}%")
                
                # Full Data Logging for Debugging
                print(f"  [OBS]: {np.round(obs, 3).tolist()}")
                print(f"  [ACT]: {np.round(action, 3).tolist()}")

            if terminated or truncated:
                print("Test Ended.")
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["hover", "pitch", "roll", "yaw"], default="hover")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    
    run_calibration(mode=args.mode, no_render=args.no_render)
