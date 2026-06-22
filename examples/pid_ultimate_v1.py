
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
print(f"DEBUG: rocket_lander imported from: {rocket_lander.__file__}")

# --- Mission State Machine ---
class MissionPhase(Enum):
    RECOVERY = 1    # Stabilize flip/tumble
    TRANSIT  = 2    # Long range quintic trajectory
    APPROACH = 3    # Final 50m precision corridor
    LANDING  = 4    # Touchdown and engine cutoff

# --- Math Utilities ---
class QuinticPolynomial:
    def __init__(self, xi, vi, ai, xf, vf, af, T):
        self.a0, self.a1, self.a2 = xi, vi, 0.5 * ai
        A = np.array([[T**3, T**4, T**5], [3*T**2, 4*T**3, 5*T**4], [6*T, 12*T**2, 20*T**3]])
        b = np.array([xf - self.a0 - self.a1*T - self.a2*T**2, vf - self.a1 - 2*self.a2*T, af - 2*self.a2])
        x = np.linalg.solve(A, b)
        self.a3, self.a4, self.a5 = x[0], x[1], x[2]
    def calc(self, t):
        xt = self.a0 + self.a1*t + self.a2*t**2 + self.a3*t**3 + self.a4*t**4 + self.a5*t**5
        vt = self.a1 + 2*self.a2*t + 3*self.a3*t**2 + 4*self.a4*t**3 + 5*self.a5*t**4
        at = 2*self.a2 + 6*self.a3*t + 12*self.a4*t**2 + 20*self.a5*t**3
        return xt, vt, at

class Trajectory3D:
    def __init__(self, start_p, start_v, end_p, end_v, duration):
        self.poly = [QuinticPolynomial(start_p[i], start_v[i], 0, end_p[i], end_v[i], 0, duration) for i in range(3)]
        self.duration = duration
    def get(self, t):
        t = np.clip(t, 0, self.duration)
        results = [p.calc(t) for p in self.poly]
        return np.array([r[0] for r in results]), np.array([r[1] for r in results]), np.array([r[2] for r in results])

class PID:
    def __init__(self, kp, ki, kd, dt, limit=255.0):
        self.kp, self.ki, self.kd, self.dt, self.limit = kp, ki, kd, dt, limit
        self.pe, self.integral = 0.0, 0.0
    def update(self, e):
        self.integral = np.clip(self.integral + e * self.dt, -20, 20)
        out = self.kp*e + self.ki*self.integral + self.kd*(e - self.pe)/self.dt
        self.pe = e
        return np.clip(out, -self.limit, self.limit)

# --- Controller ---
class UltimateController:
    def __init__(self, dt=1./60.):
        self.dt = dt
        self.phase = MissionPhase.RECOVERY
        self.pwm_mid = 127.5
        self.gravity_ff = 10.0 # Adjusted for mass
        
        # PIDs tuned for PWM Scale (Outputs -255 to 255)
        # Note: Gains increased to handle new COM stability
        self.roll_pwm_pid  = PID(500.0, 20.0, 800.0, dt)
        self.pitch_pwm_pid = PID(500.0, 20.0, 800.0, dt)
        self.yaw_pwm_pid   = PID(150.0, 0.0, 300.0, dt)
        
        self.vz_pwm_pid    = PID(25.0, 5.0, 10.0, dt)
        self.vx_pwm_pid    = PID(3.0, 0.1, 2.0, dt, limit=120) 
        self.vy_pwm_pid    = PID(3.0, 0.1, 2.0, dt, limit=120)
        
        self.history = []
        self.traj = None
        self.start_t = 0

    def compute(self, obs, time_now):
        # Auto-detect if scaled or raw
        # If absolute pos > 10.0, it's likely raw meters (since max normalized is usually 2-5)
        raw_mode = np.any(np.abs(obs[0:3]) > 10.0)
        
        if not raw_mode:
            pos_raw = obs[0:3] * 500.0
            vel_raw = obs[7:10] * 50.0
            ang_vel_raw = obs[10:13] * 10.0
            alt_raw = obs[17] * 500.0
        else:
            pos_raw = obs[0:3]
            vel_raw = obs[7:10]
            ang_vel_raw = obs[10:13]
            alt_raw = obs[17]
            
        quat = obs[3:7]
        cp, cv, alt = pos_raw, vel_raw, alt_raw
        r, p_ang, y = p.getEulerFromQuaternion(quat)
        dist_h = np.linalg.norm(cp[0:2])
        
        # 0. Mass Estimation
        fuel_pct = obs[18]
        curr_m = 300.0 + (2700.0 * fuel_pct) 
        k_mass = curr_m / 3000.0
        
        # 1. Mission State Logic
        if self.phase == MissionPhase.RECOVERY:
            if abs(r) < 0.2 and abs(p_ang) < 0.2 and np.linalg.norm(cv[0:2]) < 5.0:
                print(f"[{time_now:.2f}s] Phase -> TRANSIT. Lock.")
                self.phase = MissionPhase.TRANSIT
                self.start_t = time_now
                # Dynamic Duration based on distance (Allows time for long-range recovery)
                flight_time = max(60.0, dist_h / 10.0 + 30.0) 
                self.traj = Trajectory3D(cp, cv, [0,0,15.0], [0,0,-0.5], duration=flight_time)
                
        elif self.phase == MissionPhase.TRANSIT:
            if dist_h < 5.0 and alt < 100.0: self.phase = MissionPhase.APPROACH
        elif self.phase == MissionPhase.APPROACH:
            if alt < 12.0 and dist_h < 1.0: self.phase = MissionPhase.LANDING

        # 2. Reference Selection & Guidance
        target_y = math.atan2(-cp[1], -cp[0])
        
        if self.phase == MissionPhase.RECOVERY:
            # Hold current position (or drift slowly) but kill velocity
            ref_p, ref_v = np.array([cp[0], cp[1], cp[2]]), np.array([0, 0, 0])
        elif self.phase == MissionPhase.TRANSIT:
            ref_p, ref_v, _ = self.traj.get(time_now - self.start_t + 1.0) 
        else: # APPROACH & LANDING
            ref_p, ref_v = np.array([0,0,0]), np.array([0,0,-1.0 if self.phase == MissionPhase.APPROACH else -0.4])

        # 3. Masterpiece Guidance (High Gain)
        kp_pos, kd_vel = 2.0, 3.0
        if self.phase == MissionPhase.TRANSIT: kp_pos, kd_vel = 1.0, 1.5
        
        ax = kp_pos * (ref_p[0] - cp[0]) + kd_vel * (ref_v[0] - cv[0])
        ay = kp_pos * (ref_p[1] - cp[1]) + kd_vel * (ref_v[1] - cv[1])
        
        # World -> Local Acceleration
        cy, sy = math.cos(y), math.sin(y)
        ah =  ax*cy + ay*sy  
        al = -ax*sy + ay*cy  
        
        # Mapping: Forward Ref -> ah+ -> target_p- (Nose Down).
        #          Left Ref -> al+ -> target_r+ (Body Left/Nose Left).
        max_tilt = 1.0 # 60 deg tilt allowed for Recovery
        # Mapping: Forward Ref -> ah+ -> target_p- (Nose Down).
        #          Left Ref -> al+ -> target_r+ (Body Left/Nose Left).
        # Mapping: target_p = -ah (Forward Force). target_r = -al (Left Force).
        
        max_tilt = 1.0 # 60 deg tilt allowed for Recovery
        if self.phase == MissionPhase.TRANSIT: max_tilt = 0.5
        
        tp_raw = np.clip(-ah * 1.5, -max_tilt, max_tilt)
        tr_raw = np.clip(-al * 1.5, -max_tilt, max_tilt)
        
        # Smoothing
        if not hasattr(self, 'target_p_smooth'): self.target_p_smooth = 0.0
        if not hasattr(self, 'target_r_smooth'): self.target_r_smooth = 0.0
        alpha = 0.3
        self.target_p_smooth = alpha * tp_raw + (1 - alpha) * self.target_p_smooth
        self.target_r_smooth = alpha * tr_raw + (1 - alpha) * self.target_r_smooth
        
        # 4. Control Loops: STABLE Inner Loop
        self.pitch_pwm_pid.kp = 5000.0 * k_mass
        self.roll_pwm_pid.kp  = 15000.0 * k_mass # Roll needs more gain usually
        kd_ang = 4000.0 * k_mass
        
        y_err = (target_y - y + math.pi) % (2*math.pi) - math.pi
        
        p_pwm = self.pitch_pwm_pid.update(self.target_p_smooth - p_ang) - kd_ang * ang_vel_raw[1]
        r_pwm = self.roll_pwm_pid.update(self.target_r_smooth - r) - kd_ang * ang_vel_raw[0]
        y_pwm = self.yaw_pwm_pid.update(y_err) - kd_ang * ang_vel_raw[2]
        
        vz_pwm = self.vz_pwm_pid.update(ref_v[2] - cv[2])
        throttle_pwm = self.pwm_mid + vz_pwm
        
        # Assemble Action
        action = np.zeros(16)
        action[3:15] = -1.0 
        action[0] = np.clip((throttle_pwm - 127.5)/127.5, -0.5, 1) # Min Throttle -0.5
        action[1] = np.clip((-p_pwm)/127.5, -1, 1) 
        action[2] = np.clip((-r_pwm)/127.5, -1, 1) 
        
        if abs(y_pwm) > 30: 
            if y_pwm > 0: action[11] = action[12] = 1.0 # Quad RCS for Yaw
            else: action[13] = action[14] = 1.0
        
        action[15] = 1.0 if alt < 25 else -1.0 
        
        # Diagnostics
        if len(self.history) % 60 == 0:
            print(f"[{time_now:.1f}s] {self.phase.name} DH: {dist_h:.1f} CP: {cp[0:2]} T_Y: {target_y:.2f}")
            print(f"      AX/AY: {ax:.1f}/{ay:.1f} AH/AL: {ah:.1f}/{al:.1f} TP/TR: {self.target_p_smooth:.2f}/{self.target_r_smooth:.2f}")
            print(f"      RPY: [{r:.2f}, {p_ang:.2f}, {y:.2f}] Cmd_Vel: {ref_v} Cur_Vel: {cv}")

        self.history.append({"t": time_now, "cp": np.copy(cp), "rp": np.copy(ref_p), "alt": alt, "phase": self.phase.name})
        return action

    def plot(self):
        if not self.history: return
        t = [h["t"] for h in self.history]
        cp = np.array([h["cp"] for h in self.history])
        rp = np.array([h["rp"] for h in self.history])
        
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))
        # Horizontal error
        err_h = np.linalg.norm(cp[:,0:2] - rp[:,0:2], axis=1)
        axs[0].plot(t, err_h); axs[0].set_title("Horizontal Tracking Error (m)"); axs[0].grid()
        # 3D Path
        axs[1].plot(cp[:,0], cp[:,1], label="Actual"); axs[1].plot(rp[:,0], rp[:,1], '--', label="Reference")
        axs[1].set_title("Top Down View (XY)"); axs[1].legend(); axs[1].axis("equal"); axs[1].grid()
        plt.savefig("ultimate_mission_report.png")
        print("Telemetry saved to ultimate_mission_report.png")

def main():
    env = gym.make("RocketLander-v0", render_mode="human", normalize_obs=False)
    
    # CRITICAL: Disable spawn randomization to force 250m start
    env.unwrapped.randomize_spawn = False
    print(f"DEBUG: env.unwrapped.randomize_spawn = {env.unwrapped.randomize_spawn}")
    
    # 1. Extreme Initial Conditions
    initial_pos = [250, 250, 250]
    initial_orn = p.getQuaternionFromEuler([math.radians(30), math.radians(-30), 0])
    obs, _ = env.reset(options={"initial_pos": initial_pos, "initial_orn": initial_orn})
    
    ctrl = UltimateController(dt=env.unwrapped.dt)
    
    for i in range(25000): 
        t = i * env.unwrapped.dt
        action = ctrl.compute(obs, t)
        obs, reward, terminated, truncated, _ = env.step(action)
        env.render()
        
        if terminated or truncated:
            # Precision Check
            pos = obs[0:3] 
            dist_final = np.linalg.norm(pos[0:2])
            print(f"MISSION TERMINATED at T={t:.2f}s. Final Error: {dist_final:.4f}m Alt: {pos[2]:.2f}m")
            if dist_final < 0.5:
                print(">>> SUCCESS: 0.5m TOLERANCE ACHIEVED! <<<")
            else:
                print(">>> FAIL: Target precision not reached. <<<")
            break
            
    ctrl.plot()
    env.close()

if __name__ == "__main__":
    main()
