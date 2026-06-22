
import gymnasium as gym
import numpy as np
import pybullet as p
import math
import time
import argparse
import sys
import os
import matplotlib.pyplot as plt
from enum import Enum
from typing import Optional, List, Tuple, Dict

# Force local import
curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(curr_dir)
sys.path.insert(0, root_dir)

import rocket_lander

class ControlState(Enum):
    RECOVERY           = 1 
    ORIENT             = 2 
    APPROACH           = 3 
    LANDING            = 4 

class PID:
    def __init__(self, kp: float, ki: float, kd: float, dt: float, limit: float = 1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.limit = limit
        self.prev_error = 0.0
        self.integral = 0.0
        
    def update(self, error: float, is_angular: bool = False) -> float:
        self.integral += error * self.dt
        self.integral = np.clip(self.integral, -2.0, 2.0) 
        
        diff = error - self.prev_error
        if is_angular:
            while diff > np.pi: diff -= 2 * np.pi
            while diff < -np.pi: diff += 2 * np.pi
            
        derivative = diff / self.dt
        self.prev_error = error
        
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return float(np.clip(output, -self.limit, self.limit))
    
    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0

class AdvancedController:
    def __init__(self, dt: float = 1.0/120.0):
        self.dt = dt
        # Hover at ~4% throttle (100kg * 9.8 / 25000)
        self.gravity_ff = 0.04 
        
        # PIDs - Tuned for stability
        self.roll_pid  = PID(kp=1.5, ki=0.05, kd=2.5, dt=dt, limit=1.0)
        self.pitch_pid = PID(kp=1.5, ki=0.05, kd=2.5, dt=dt, limit=1.0)
        self.yaw_pid   = PID(kp=0.5, ki=0.0, kd=1.0, dt=dt, limit=1.0)
        
        self.vz_pid    = PID(kp=0.1, ki=0.01, kd=0.02, dt=dt, limit=0.8) 
        self.vx_pid    = PID(kp=0.01, ki=0.0, kd=0.01, dt=dt, limit=0.1) 
        self.vy_pid    = PID(kp=0.01, ki=0.0, kd=0.01, dt=dt, limit=0.1)
        
        self.state = ControlState.RECOVERY
        self.history = []

    def reset(self):
        for pid in [self.roll_pid, self.pitch_pid, self.yaw_pid, self.vz_pid, self.vx_pid, self.vy_pid]:
            pid.reset()
        self.state = ControlState.RECOVERY
        self.history = []

    def compute_action(self, obs: np.ndarray) -> np.ndarray:
        pos = obs[0:3]
        quat = obs[3:7]
        vel = obs[7:10]
        local_ang_vel = obs[10:13]
        fuel = obs[18]
        altitude_sensor = obs[17]
        
        roll, pitch, yaw = p.getEulerFromQuaternion(quat)
        dist_h = np.linalg.norm(pos[0:2])
        
        # 1. State Machine
        if self.state == ControlState.RECOVERY:
            if abs(roll) < 0.2 and abs(pitch) < 0.2 and np.linalg.norm(local_ang_vel) < 0.5:
                self.state = ControlState.ORIENT
        elif self.state == ControlState.ORIENT:
            if dist_h > 30: self.state = ControlState.APPROACH
            else:           self.state = ControlState.LANDING
        elif self.state == ControlState.APPROACH:
            if dist_h < 15: self.state = ControlState.LANDING
        
        # 2. Guidance
        target_yaw = 0.0
        if self.state == ControlState.LANDING:
            target_vz = -2.0 if altitude_sensor > 10 else -1.0
            target_vx, target_vy = np.clip(-pos[0] * 0.4, -5, 5), np.clip(-pos[1] * 0.4, -5, 5)
        else:
            target_h = 200.0
            target_vz = np.clip((target_h - altitude_sensor) * 0.15, -20, 20)
            target_vx, target_vy = np.clip(-pos[0] * 0.1, -15, 15), np.clip(-pos[1] * 0.1, -15, 15)

        # 3. PID Cascades
        # Horizontal Control -> Tilt Targets (rad)
        # Transform velocity error to local frame
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        err_x_world = target_vx - vel[0]
        err_y_world = target_vy - vel[1]
        err_x_local = err_x_world * cos_y + err_y_world * sin_y
        err_y_local = -err_x_world * sin_y + err_y_world * cos_y
        
        target_pitch = self.vx_pid.update(err_x_local)
        target_roll = -self.vy_pid.update(err_y_local)
        
        # Attitude -> RCS
        p_cmd = self.pitch_pid.update(target_pitch - pitch)
        r_cmd = self.roll_pid.update(target_roll - roll)
        y_cmd = self.yaw_pid.update(target_yaw - yaw, is_angular=True)
        
        # Vertical -> Throttle
        throttle_out = self.vz_pid.update(target_vz - vel[2])
        throttle = np.clip(self.gravity_ff + throttle_out, 0.0, 1.0)
        
        # 4. Action Mapping (16-dim)
        action = np.zeros(16)
        action[0] = throttle * 2.0 - 1.0 # Throttle [-1, 1]
        action[1] = np.clip(p_cmd, -1, 1) # Gimbal P
        action[2] = np.clip(r_cmd, -1, 1) # Gimbal R
        
        # Proportional RCS Logic
        # Top [3:7]: X+, X-, Y+, Y-
        if p_cmd > 0: action[4] = p_cmd # Top X- pushes +X -> Pitch Up
        else:         action[3] = -p_cmd # Top X+ pushes -X -> Pitch Down
        
        if r_cmd > 0: action[6] = r_cmd  # Top Y- pushes +Y -> Roll
        else:         action[5] = -r_cmd # Top Y+ pushes -Y
        
        # Bottom [7:11]: X+, X-, Y+, Y- (Reverse lever arm)
        if p_cmd > 0: action[7] = p_cmd  # Bottom X+ pushes -X -> Pitch Up
        else:         action[8] = -p_cmd 
        
        if r_cmd > 0: action[9] = r_cmd  # Bottom Y+ pushes -Y -> Roll
        else:         action[10] = -r_cmd
        
        # Mid [11:15]: Yaw (Tangent)
        if y_cmd > 0: action[11] = y_cmd; action[14] = y_cmd
        else:         action[12] = -y_cmd; action[13] = -y_cmd
            
        action[15] = 1.0 if (self.state == ControlState.LANDING and altitude_sensor < 60) else -1.0
        
        # History
        self.history.append({
            "t": len(self.history) * self.dt,
            "pos": np.copy(pos), "vel": np.copy(vel), "rpy": [roll, pitch, yaw],
            "throttle": throttle, "fuel": fuel, "state": self.state.name, "alt": altitude_sensor
        })
        return action

    def plot_mission(self, save_path="mission_report.png"):
        if not self.history: return
        t = [h["t"] for h in self.history]
        alt = [h["alt"] for h in self.history]
        vx = [h["vel"][0] for h in self.history]
        vy = [h["vel"][1] for h in self.history]
        vz = [h["vel"][2] for h in self.history]
        fuel = [h["fuel"] for h in self.history]
        
        fig, axs = plt.subplots(3, 2, figsize=(12, 12))
        axs[0, 0].plot(t, alt); axs[0, 0].set_title("Altitude (m)"); axs[0, 0].grid()
        axs[0, 1].plot(t, vz); axs[0, 1].set_title("Vertical Velocity (m/s)"); axs[0, 1].grid()
        axs[1, 0].plot(t, vx, label="Vx"); axs[1, 0].plot(t, vy, label="Vy"); axs[1, 0].set_title("Horizontal Velocity"); axs[1, 0].legend(); axs[1, 0].grid()
        
        x = [h["pos"][0] for h in self.history]; y = [h["pos"][1] for h in self.history]
        axs[1, 1].plot(x, y); axs[1, 1].set_title("Ground Track (X vs Y)"); axs[1, 1].axis("equal"); axs[1, 1].grid()
        
        r = [math.degrees(h["rpy"][0]) for h in self.history]; p_deg = [math.degrees(h["rpy"][1]) for h in self.history]
        axs[2, 0].plot(t, r, label="Roll"); axs[2, 0].plot(t, p_deg, label="Pitch"); axs[2, 0].set_title("Attitude (deg)"); axs[2, 0].legend(); axs[2, 0].grid()
        
        axs[2, 1].plot(t, [f * 100 for f in fuel]); axs[2, 1].set_title("Fuel (%)"); axs[2, 1].grid()
        
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"DEBUG: Mission report saved to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spawn", type=str, default="0,0,500", help="x,y,z spawn position")
    parser.add_argument("--rpy", type=str, default="0,0,0", help="r,p,y spawn orientation (deg)")
    parser.add_argument("--plot", action="store_true", help="Generate plot")
    args = parser.parse_args()
    
    try:
        sp = [float(x) for x in args.spawn.split(",")]
        or_rot = [math.radians(float(x)) for x in args.rpy.split(",")]
    except:
        print("ERROR: Invalid format")
        return

    env = gym.make("RocketLander-v0", render_mode="human")
    ctrl = AdvancedController()
    obs, _ = env.reset(options={"initial_pos": sp, "initial_orn": p.getQuaternionFromEuler(or_rot)})
    
    print(f"DEBUG: Final PID Test Started. Spawn={sp}")
    
    for _ in range(15000):
        action = ctrl.compute_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
            
    if args.plot: ctrl.plot_mission()
    env.close()

if __name__ == "__main__":
    main()
