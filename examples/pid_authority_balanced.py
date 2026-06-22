
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
    def __init__(self, kp: float, ki: float, kd: float, dt: float, limit: float = 255.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.limit = limit # Internal computation in PWM (0-255 range for magnitude)
        self.prev_error = 0.0
        self.integral = 0.0
        
    def update(self, error: float, is_angular: bool = False) -> float:
        self.integral += error * self.dt
        self.integral = np.clip(self.integral, -10.0, 10.0) 
        
        diff = error - self.prev_error
        if is_angular:
            while diff > np.pi: diff -= 2 * np.pi
            while diff < -np.pi: diff += 2 * np.pi
            
        derivative = diff / self.dt
        self.prev_error = error
        
        # Output is in "PWM" scale (-limit to +limit)
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return float(np.clip(output, -self.limit, self.limit))
    
    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0

class AuthorityController:
    def __init__(self, dt: float = 1.0/120.0):
        self.dt = dt
        # Internal PWM bias (Mid-point is 127.5)
        self.pwm_mid = 127.5
        
        # Hover FF in PWM: ~4% of 255 = ~10 units
        # (100kg * 9.8 / 25000) * 255
        self.gravity_ff_pwm = 10.0 
        
        # PIDs - Tuned for PWM Scale (outputs -255 to 255)
        # Note: Gains are scaled to map physical errors to the 0-255 range
        self.roll_pwm_pid  = PID(kp=400.0, ki=10.0, kd=600.0, dt=dt)
        self.pitch_pwm_pid = PID(kp=400.0, ki=10.0, kd=600.0, dt=dt)
        self.yaw_pwm_pid   = PID(kp=100.0, ki=0.0, kd=200.0, dt=dt)
        
        self.vz_pwm_pid    = PID(kp=20.0, ki=2.0, kd=5.0, dt=dt) 
        self.vx_pwm_pid    = PID(kp=2.0, ki=0.0, kd=1.0, dt=dt, limit=100.0) # Horizontal tilt limit (PWM)
        self.vy_pwm_pid    = PID(kp=2.0, ki=0.0, kd=1.0, dt=dt, limit=100.0)
        
        self.state = ControlState.RECOVERY
        self.history = []

    def normalize_pwm(self, pwm_val: float) -> float:
        """Converts internal PWM (0-255) to gym action (-1.0 to 1.0)"""
        # (pwm_val - 127.5) / 127.5
        return float(np.clip((pwm_val - self.pwm_mid) / self.pwm_mid, -1.0, 1.0))

    def compute_action(self, obs: np.ndarray) -> np.ndarray:
        pos = obs[0:3]
        quat = obs[3:7]
        vel = obs[7:10]
        local_ang_vel = obs[10:13]
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
        
        # 2. AUTHORITY SCALING (The "Secret Sauce")
        # Closer to pad = tighter constraints, Far away = aggressive authority
        # Scale tilt limit from 0.4 rad (far) to 0.05 rad (near)
        authority = np.clip(1.0 - dist_h / 100.0, 0.0, 1.0) # 0 if far, 1 if near
        max_tilt = 0.4 - (0.35 * authority) 
        
        # 3. Guidance logic
        target_yaw = 0.0
        if self.state == ControlState.LANDING:
            target_vz = -np.clip(0.15 * altitude_sensor, 0.5, 2.0)
            target_vx, target_vy = np.clip(-pos[0] * 0.4, -2.0, 2.0) , np.clip(-pos[1] * 0.4, -2.0, 2.0)
        else:
            # Aggressive horizontal target when far
            target_speed = np.clip(0.15 * dist_h, 2.0, 15.0)
            target_vz = -2.0 if altitude_sensor > 50 else -1.0
            
            if dist_h > 0.1:
                dir_to_pad = -pos[0:2] / dist_h
                target_vx = dir_to_pad[0] * target_speed
                target_vy = dir_to_pad[1] * target_speed
            else:
                target_vx = target_vy = 0.0

        # 4. PID Cascades (Internal PWM Math)
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        err_x_world = target_vx - vel[0]
        err_y_world = target_vy - vel[1]
        err_x_local = err_x_world * cos_y + err_y_world * sin_y
        err_y_local = -err_x_world * sin_y + err_y_world * cos_y
        
        # Horizontal velocity to Tilt targets (limit by authority)
        target_pitch = np.clip(self.vx_pwm_pid.update(err_x_local) / 255.0, -max_tilt, max_tilt)
        target_roll = np.clip(-self.vy_pwm_pid.update(err_y_local) / 255.0, -max_tilt, max_tilt)
        
        # Attitude commands in PWM
        p_pwm_raw = self.pitch_pwm_pid.update(target_pitch - pitch)
        r_pwm_raw = self.roll_pwm_pid.update(target_roll - roll)
        y_pwm_raw = self.yaw_pwm_pid.update(target_yaw - yaw, is_angular=True)
        
        # Vertical command in PWM
        vz_pwm_raw = self.vz_pwm_pid.update(target_vz - vel[2])
        throttle_pwm = self.pwm_mid + self.gravity_ff_pwm + vz_pwm_raw
        
        # 5. Final Action Mapping (0-255 -> -1 to 1)
        action = np.zeros(16)
        action[0] = self.normalize_pwm(throttle_pwm)
        
        # Gimbal P/R (Centralized)
        action[1] = self.normalize_pwm(self.pwm_mid + p_pwm_raw)
        action[2] = self.normalize_pwm(self.pwm_mid + r_pwm_raw)
        
        # RCS Logic (P/R) - Use PWM raw for magnitude
        # Mapping remains proportional but derived from internal PWM limits
        def apply_rcs(cmd_pwm, pos_idx, neg_idx):
            if cmd_pwm > 10:  action[pos_idx] = cmd_pwm / 255.0
            if cmd_pwm < -10: action[neg_idx] = -cmd_pwm / 255.0

        apply_rcs(p_pwm_raw, 4, 3) # top
        apply_rcs(r_pwm_raw, 6, 5) # top
        apply_rcs(p_pwm_raw, 7, 8) # bottom (reverse)
        apply_rcs(r_pwm_raw, 9, 10) # bottom (reverse)
        
        # Yaw RCS
        if y_pwm_raw > 10:  action[11] = action[14] = y_pwm_raw / 255.0
        if y_pwm_raw < -10: action[12] = action[13] = -y_pwm_raw / 255.0
            
        action[15] = 1.0 if (self.state == ControlState.LANDING and altitude_sensor < 60) else -1.0
        
        # 6. Logging
        self.history.append({
            "t": len(self.history) * self.dt,
            "pos": np.copy(pos), "vel": np.copy(vel), "rpy": [roll, pitch, yaw],
            "throttle": self.normalize_pwm(throttle_pwm), "state": self.state.name, "alt": altitude_sensor
        })
        return action

    def plot_mission(self, save_path="pid_authority_report.png"):
        if not self.history: return
        t = [h["t"] for h in self.history]
        alt = [h["alt"] for h in self.history]
        vz = [h["vel"][2] for h in self.history]
        vx = [h["vel"][0] for h in self.history]
        vy = [h["vel"][1] for h in self.history]
        
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        axs[0, 0].plot(t, alt); axs[0, 0].set_title("Altitude (m)"); axs[0, 0].grid()
        axs[0, 1].plot(t, vz); axs[0, 1].set_title("Vz (m/s)"); axs[0, 1].grid()
        axs[1, 0].plot(t, vx, label="Vx"); axs[1, 0].plot(t, vy, label="Vy"); axs[1, 0].set_title("Horizontal Vel"); axs[1, 0].legend(); axs[1, 0].grid()
        
        x = [h["pos"][0] for h in self.history]; y = [h["pos"][1] for h in self.history]
        axs[1, 1].plot(x, y); axs[1, 1].set_title("Ground Track"); axs[1, 1].axis("equal"); axs[1, 1].grid()
        
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"DEBUG: Mission report saved to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spawn_radius", type=float, default=200.0)
    args = parser.parse_args()

    env = gym.make("RocketLander-v0", render_mode="human")
    ctrl = AuthorityController()
    
    # Test random spawns
    obs, _ = env.reset(options={"spawn_radius": args.spawn_radius})
    print(f"DEBUG: Authority-Balanced PID Test Started. Difficulty={args.spawn_radius}m")
    
    for _ in range(20000):
        action = ctrl.compute_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        env.render()
        if terminated or truncated:
            break
            
    ctrl.plot_mission()
    env.close()

if __name__ == "__main__":
    main()
