
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

# --- Poly Trajectory Classes ---

class QuinticPolynomial:
    def __init__(self, xi, vi, ai, xf, vf, af, T):
        self.a0 = xi
        self.a1 = vi
        self.a2 = 0.5 * ai
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
        self.Tx = QuinticPolynomial(start_p[0], start_v[0], 0, end_p[0], end_v[0], 0, duration)
        self.Ty = QuinticPolynomial(start_p[1], start_v[1], 0, end_p[1], end_v[1], 0, duration)
        self.Tz = QuinticPolynomial(start_p[2], start_v[2], 0, end_p[2], end_v[2], 0, duration)
        self.duration = duration

    def get_state(self, t):
        t = np.clip(t, 0, self.duration)
        rx, vx, ax = self.Tx.calc(t)
        ry, vy, ay = self.Ty.calc(t)
        rz, vz, az = self.Tz.calc(t)
        return np.array([rx, ry, rz]), np.array([vx, vy, vz]), np.array([ax, ay, az])

# --- Controller Classes ---

class PID:
    def __init__(self, kp, ki, kd, dt, limit=255.0):
        self.kp, self.ki, self.kd, self.dt, self.limit = kp, ki, kd, dt, limit
        self.pe, self.integral = 0.0, 0.0
    def update(self, e):
        self.integral = np.clip(self.integral + e * self.dt, -10, 10)
        out = self.kp*e + self.ki*self.integral + self.kd*(e - self.pe)/self.dt
        self.pe = e
        return np.clip(out, -self.limit, self.limit)

class TrajectoryController:
    def __init__(self, dt=1./60.):
        self.dt = dt
        self.pwm_mid = 127.5
        # FF for gravity in PWM (~10 units)
        self.gravity_ff = 10.0
        
        # Inner Loop: Attitude (PWM)
        self.roll_pid = PID(400, 10, 600, dt)
        self.pitch_pid = PID(400, 10, 600, dt)
        self.yaw_pid = PID(100, 0, 200, dt)
        
        # Outer Loop: Velocity -> Tilt Target
        self.vx_pid = PID(2, 0, 1, dt, limit=100)
        self.vy_pid = PID(2, 0, 1, dt, limit=100)
        self.vz_pid = PID(20, 2, 5, dt)
        
        self.history = []

    def compute(self, obs, ref_p, ref_v, ref_a):
        cp, quat, cv, alt = obs[0:3], obs[3:7], obs[7:10], obs[17]
        r, p_ang, y = p.getEulerFromQuaternion(quat)
        dist_h = np.linalg.norm(cp[0:2])
        
        # Authority Scaling: Relax constraints when far
        authority = np.clip(1.0 - dist_h / 50.0, 0.0, 1.0) # 0 far, 1 near
        max_tilt = 0.4 - (0.35 * authority) 
        
        # 1. Outer Loop (Cascaded)
        # Transform error to body frame
        cy, sy = math.cos(y), math.sin(y)
        ev_w = ref_v - cv
        ev_l = [ev_w[0]*cy + ev_w[1]*sy, -ev_w[0]*sy + ev_w[1]*cy]
        
        target_p = np.clip(self.vx_pid.update(ev_l[0]) / 255.0, -max_tilt, max_tilt)
        target_r = np.clip(-self.vy_pid.update(ev_l[1]) / 255.0, -max_tilt, max_tilt)
        
        # 2. Inner Loop (Attitude)
        p_pwm = self.pitch_pid.update(target_p - p_ang)
        r_pwm = self.roll_pid.update(target_r - r)
        y_pwm = self.yaw_pid.update(0 - y)
        
        # 3. Vertical (Throttle)
        vz_pwm = self.vz_pid.update(ref_v[2] - cv[2])
        throttle_pwm = self.pwm_mid + self.gravity_ff + vz_pwm
        
        # 4. Mapper (0-255 -> -1..1)
        action = np.zeros(16)
        action[0] = np.clip((throttle_pwm - 127.5)/127.5, -1, 1)
        action[1] = np.clip((self.pwm_mid + p_pwm - 127.5)/127.5, -1, 1)
        action[2] = np.clip((self.pwm_mid + r_pwm - 127.5)/127.5, -1, 1)
        
        # RCS Logic
        def set_rcs(val, p_idx, n_idx):
            if val > 10: action[p_idx] = val/255.0
            elif val < -10: action[n_idx] = -val/255.0
        set_rcs(p_pwm, 4, 3); set_rcs(r_pwm, 6, 5) # top
        set_rcs(p_pwm, 7, 8); set_rcs(r_pwm, 9, 10) # bottom
        if y_pwm > 10: action[11] = action[14] = y_pwm/255.0
        elif y_pwm < -10: action[12] = action[13] = -y_pwm/255.0
        
        action[15] = 1.0 if alt < 50 else -1.0 # Legs
        
        self.history.append({"t": len(self.history)*self.dt, "cp": cp, "rp": ref_p, "alt": alt})
        return action

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spawn", type=str, default="50,50,100")
    args = parser.parse_args()
    sp = [float(x) for x in args.spawn.split(",")]

    env = gym.make("RocketLander-v0", render_mode="human")
    obs, _ = env.reset(options={"initial_pos": sp})
    
    ctrl = TrajectoryController(dt=env.unwrapped.dt)
    duration = 30.0 # 30 seconds to land
    traj = Trajectory3D(sp, [0,0,0], [0,0,0], [0,0,-0.5], duration)
    
    print(f"MISSION START: Trajectory Trapping (Targeting Landing Pad in {duration}s)")
    
    for i in range(10000):
        t = i * env.unwrapped.dt
        ref_p, ref_v, ref_a = traj.get_state(t)
        action = ctrl.compute(obs, ref_p, ref_v, ref_a)
        obs, _, done, trunc, _ = env.step(action)
        env.render()
        if done or trunc: break
        
    env.close()

if __name__ == "__main__":
    main()
