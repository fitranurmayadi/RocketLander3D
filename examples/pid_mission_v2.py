import gymnasium as gym
import rocket_lander
import numpy as np
import pybullet as p
import math
import time
import os
import matplotlib.pyplot as plt
from enum import Enum
from typing import Dict, Any, Tuple, Optional

class ControlState(Enum):
    RECOVERY    = 1 # Stabilize tumble
    APPROACH    = 2 # Fly to waypoint (0, 0, 100)
    ALIGN_NORTH = 3 # Face North (Yaw 0)
    LANDING     = 4 # Controlled descent to pad

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
        self.integral = np.clip(self.integral, -self.limit, self.limit)
        
        diff = error - self.prev_error
        if is_angular:
            while diff > math.pi: diff -= 2 * math.pi
            while diff < -math.pi: diff += 2 * math.pi
            
        derivative = diff / self.dt
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative
    
    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0

class StateMachineController:
    def __init__(self, dt: float = 1.0/30.0):
        self.dt = dt
        # Tuning for 5000N Engine and 100N RCS
        self.roll_pid  = PID(kp=4.0, ki=0.01, kd=15.0, dt=dt)
        self.pitch_pid = PID(kp=4.0, ki=0.01, kd=15.0, dt=dt)
        self.yaw_pid   = PID(kp=2.0, ki=0.01, kd=5.0,  dt=dt)
        self.vz_pid    = PID(kp=0.5, ki=0.05, kd=0.1,  dt=dt) # Throttle control
        self.vx_pid    = PID(kp=1.5, ki=0.0,  kd=0.5,  dt=dt) # RCS Translation
        self.vy_pid    = PID(kp=1.5, ki=0.0,  kd=0.5,  dt=dt)
        
        self.gravity_ff = 0.44 # Approx hover throttle for 5000N (~220kg)
        
        self.reset()
        
    def reset(self):
        for p in [self.roll_pid, self.pitch_pid, self.yaw_pid, self.vz_pid, self.vx_pid, self.vy_pid]:
            p.reset()
        self.state = ControlState.RECOVERY
        self.step_count = 0
        self.attitude_stable_count = 0
        
    def normalize_angle(self, angle: float) -> float:
        while angle > math.pi: angle -= 2 * math.pi
        while angle < -math.pi: angle += 2 * math.pi
        return angle

    def compute_action(self, obs: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        # Obs: pos(3), quat(4), vel(3), ang_vel(3), contacts(4), alt(1), fuel(1)
        # Total 19
        pos = obs[0:3]
        quat = obs[3:7]
        vel = obs[7:10]
        ang_vel = obs[10:13] # Local frame
        fuel = obs[18]
        
        r, p, y = p.getEulerFromQuaternion(quat)
        alt = obs[17]
        dist_h = np.linalg.norm(pos[0:2])
        
        # 1. State Machine Transitions
        is_stable = abs(r) < 0.1 and abs(p) < 0.1
        if is_stable: self.attitude_stable_count += 1
        else: self.attitude_stable_count = 0
        
        if self.state == ControlState.RECOVERY:
            if self.attitude_stable_count > 30: # 1s stable
                self.state = ControlState.APPROACH
        
        elif self.state == ControlState.APPROACH:
            if dist_h < 5.0 and abs(alt - 50.0) < 5.0:
                self.state = ControlState.ALIGN_NORTH
        
        elif self.state == ControlState.ALIGN_NORTH:
            if abs(self.normalize_angle(y)) < 0.1:
                self.state = ControlState.LANDING
        
        # 2. Control Logic
        target_vz = 0.0
        target_yaw = 0.0
        target_vx = 0.0
        target_vy = 0.0
        deploy_legs = -1.0
        
        if self.state == ControlState.RECOVERY:
            target_vz = 0.0
            target_yaw = y
            target_vx, target_vy = 0.0, 0.0
            
        elif self.state == ControlState.APPROACH:
            # Fly to waypoint (0, 0, 50)
            target_vz = np.clip((50.0 - alt) * 0.2, -10.0, 10.0)
            target_vx = np.clip(-pos[0] * 0.1, -5.0, 5.0)
            target_vy = np.clip(-pos[1] * 0.1, -5.0, 5.0)
            target_yaw = math.atan2(-pos[1], -pos[0]) if dist_h > 2.0 else 0.0

        elif self.state == ControlState.ALIGN_NORTH:
            target_vz = np.clip((50.0 - alt) * 0.2, -5.0, 5.0)
            target_yaw = 0.0
            target_vx = np.clip(-pos[0] * 0.2, -2.0, 2.0)
            target_vy = np.clip(-pos[1] * 0.2, -2.0, 2.0)

        elif self.state == ControlState.LANDING:
            target_yaw = 0.0
            target_vx = np.clip(-pos[0] * 0.3, -2.0, 2.0)
            target_vy = np.clip(-pos[1] * 0.3, -2.0, 2.0)
            
            if alt > 15: target_vz = -5.0
            elif alt > 5: target_vz = -2.0
            else: target_vz = -0.8
            
            deploy_legs = 1.0

        # 3. PID Execution
        action = np.full(16, -1.0)
        
        # Vertical (Main Engine)
        vz_err = target_vz - vel[2]
        action[0] = np.clip(self.gravity_ff + self.vz_pid.update(vz_err), -1, 1)
        
        # Attitude (RCS Pitch/Roll/Yaw)
        # Using negative angles as error to stabilize to 0
        r_cmd = self.roll_pid.update(-r)
        p_cmd = self.pitch_pid.update(-p)
        y_err = self.normalize_angle(target_yaw - y)
        y_cmd = self.yaw_pid.update(y_err, is_angular=True)

        # Map RPY to indices
        if p_cmd > 0.1: action[4]=action[7] = min(1.0, abs(p_cmd))
        elif p_cmd < -0.1: action[3]=action[8] = min(1.0, abs(p_cmd))
        
        if r_cmd > 0.1: action[5]=action[10] = min(1.0, abs(r_cmd))
        elif r_cmd < -0.1: action[6]=action[9] = min(1.0, abs(r_cmd))
        
        if y_cmd > 0.1: action[11]=action[12] = min(1.0, abs(y_cmd))
        elif y_cmd < -0.1: action[13]=action[14] = min(1.0, abs(y_cmd))

        # Translation (RCS X/Y)
        # Convert world-frame velocity targets to local frame
        # (Simplified: only fire when reasonably level)
        if abs(r) < 0.3 and abs(p) < 0.3:
            vx_body = target_vx - vel[0]
            vy_body = target_vy - vel[1]
            # World rotation to local (Yaw inverse)
            cy, sy = math.cos(-y), math.sin(-y)
            lx = vx_body * cy - vy_body * sy
            ly = vx_body * sy + vy_body * cy
            
            tx_cmd = self.vx_pid.update(lx)
            ty_cmd = self.vy_pid.update(ly)
            
            if tx_cmd > 0.2: action[4]=action[8] = max(action[4], min(1.0, abs(tx_cmd)))
            elif tx_cmd < -0.2: action[3]=action[7] = max(action[3], min(1.0, abs(tx_cmd)))
            
            if ty_cmd > 0.2: action[6]=action[10] = max(action[6], min(1.0, abs(ty_cmd)))
            elif ty_cmd < -0.2: action[5]=action[9] = max(action[5], min(1.0, abs(ty_cmd)))

        # Legs
        action[15] = deploy_legs
        
        debug = {
            "state": self.state.name,
            "target_vz": target_vz,
            "target_yaw": math.degrees(target_yaw),
            "alt": alt,
            "dist": dist_h
        }
        self.step_count += 1
        return action, debug

def plot_mission_report(logs: Dict[str, list], filename: str):
    fig, axs = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle("Rocket Lander Mission Report (PID V2)", fontsize=16)
    
    t = np.array(logs["step"]) / 30.0
    
    # 1. Height & Vz
    axs[0, 0].plot(t, logs["alt"], 'b', label="Altitude (m)")
    axs[0, 0].set_ylabel("Altitude (m)")
    ax_vz = axs[0, 0].twinx()
    ax_vz.plot(t, logs["vz"], 'r', alpha=0.3, label="Vz (m/s)")
    ax_vz.set_ylabel("Vz (m/s)")
    axs[0, 0].legend(loc="upper left"); ax_vz.legend(loc="upper right")
    axs[0, 0].grid(True)
    
    # 2. X-Y Position
    axs[0, 1].plot(logs["x"], logs["y"], 'g')
    axs[0, 1].plot(logs["x"][0], logs["y"][0], 'go', label="Start")
    axs[0, 1].plot(0, 0, 'rx', label="Pad")
    axs[0, 1].set_xlabel("X (m)"); axs[0, 1].set_ylabel("Y (m)")
    axs[0, 1].set_title("Trajectory")
    axs[0, 1].legend(); axs[0, 1].grid(True)
    
    # 3. Attitude (Euler Angles)
    axs[1, 0].plot(t, np.degrees(logs["roll"]), label="Roll")
    axs[1, 0].plot(t, np.degrees(logs["pitch"]), label="Pitch")
    axs[1, 0].plot(t, np.degrees(logs["yaw"]), label="Yaw")
    axs[1, 0].set_ylabel("Degrees")
    axs[1, 0].legend(); axs[1, 0].grid(True); axs[1, 0].set_title("Attitude")
    
    # 4. Control Action (Throttle)
    axs[1, 1].plot(t, (np.array(logs["throttle"]) + 1.0)/2.0, 'k')
    axs[1, 1].set_ylabel("Throttle %")
    axs[1, 1].set_title("Main Engine")
    axs[1, 1].grid(True)
    
    # 5. Fuel
    axs[2, 0].plot(t, logs["fuel"], 'orange')
    axs[2, 0].set_ylabel("Fuel Remaining")
    axs[2, 0].set_title("Resource Consumption")
    axs[2, 0].grid(True)
    
    # 6. State Lifecycle
    axs[2, 1].plot(t, logs["state_id"], 'purple')
    axs[2, 1].set_ylabel("State ID")
    axs[2, 1].set_yticks([1, 2, 3, 4])
    axs[2, 1].set_yticklabels(["RECOVERY", "APPROACH", "ALIGN", "LANDING"])
    axs[2, 1].set_title("Mission Phase")
    axs[2, 1].grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename)
    print(f"Report saved to {filename}")

def main():
    env = gym.make("RocketLander-v0", render_mode="human")
    
    for ep in range(3):
        print(f"\n--- Episode {ep+1} ---")
        obs, _ = env.reset()
        ctrl = StateMachineController()
        
        logs = {
            "step": [], "alt": [], "vz": [], "x": [], "y": [],
            "roll": [], "pitch": [], "yaw": [], "throttle": [],
            "fuel": [], "state_id": []
        }
        
        done = False
        while not done:
            action, debug = ctrl.compute_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Log data
            logs["step"].append(ctrl.step_count)
            logs["alt"].append(obs[17])
            logs["vz"].append(obs[9])
            logs["x"].append(obs[0]); logs["y"].append(obs[1])
            r, p_quat, y = p.getEulerFromQuaternion(obs[3:7])
            logs["roll"].append(r); logs["pitch"].append(p_quat); logs["yaw"].append(y)
            logs["throttle"].append(action[0])
            logs["fuel"].append(obs[18])
            logs["state_id"].append(ControlState[debug["state"]].value)
            
            if ctrl.step_count % 30 == 0:
                print(f"Step {ctrl.step_count:4d} | State: {debug['state']:10s} | Alt: {debug['alt']:6.1f}m | Dist: {debug['dist']:5.1f}m")
        
        plot_mission_report(logs, f"mission_report_v2_ep{ep+1}.png")
        time.sleep(1)

    env.close()

if __name__ == "__main__":
    main()
