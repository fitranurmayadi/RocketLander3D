import gymnasium as gym
import rocket_lander
import pybullet as p
import numpy as np
import time
import math
import argparse

class PID:
    def __init__(self, kp, ki, kd, target=0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target = target
        self.integral = 0
        self.prev_val = None
        self.prev_deriv = 0.0

    def compute(self, current_val, dt):
        if self.prev_val is None:
            self.prev_val = current_val
            
        error = self.target - current_val
        self.integral += error * dt
        self.integral = np.clip(self.integral, -1.0, 1.0)
        
        # Derivative on Measurement with Low-Pass Filtering (alpha=0.3)
        raw_deriv = -(current_val - self.prev_val) / dt
        self.prev_val = current_val
        
        alpha = 0.3
        filtered_deriv = (1-alpha)*self.prev_deriv + alpha*raw_deriv
        self.prev_deriv = filtered_deriv
        
        return self.kp * error + self.ki * self.integral + self.kd * filtered_deriv

class RocketPIDController:
    def __init__(self, target_height=300.0):
        # Constants for mass estimation
        self.MIN_MASS = 300.0
        self.MAX_MASS = 3000.0
        self.FUEL_MASS = self.MAX_MASS - self.MIN_MASS
        
        # 1. Vertical Control (Throttle)
        self.alt_pid = PID(kp=0.25, ki=0.01, kd=0.1, target=target_height) 
        self.vz_pid = PID(kp=0.2, ki=0.01, kd=0.05, target=-2.0) 
        
        # 2. Horizontal Body Position Control (Dist -> Local Speed)
        # Using body-frame for more intuitive movement
        self.body_px_pid = PID(kp=0.15, ki=0.0, kd=0.02, target=0.0)
        self.body_py_pid = PID(kp=0.15, ki=0.0, kd=0.02, target=0.0)
        
        # 3. Horizontal Body Velocity Control (Local Speed -> Angle)
        # Accel Forward (+Vx) requires Nose-DOWN (-Pitch) -> Negative Kp
        self.body_vx_pid = PID(kp=-0.25, ki=-0.01, kd=-0.05, target=0.0)
        # Accel Left (+Vy) requires Tilt-LEFT (+Roll) -> Positive Kp
        self.body_vy_pid = PID(kp=0.25, ki=0.01, kd=0.05, target=0.0)
        
        # 4. Attitude Control (Gimbal/RCS)
        self.roll_pid = PID(kp=-4.0, ki=-0.0, kd=-3.0, target=0)
        self.pitch_pid = PID(kp=-2.5, ki=-0.0, kd=-3.5, target=0)
        self.yaw_pid = PID(kp=4.0, ki=0.0, kd=2.0, target=0)
        
        self.max_gimbal_deg = 20.0 # 0.35 rad
        self.max_motor_torque = 50000.0 
        self.lever_arm = 2.8 
        self.last_telem = ""
        self.smooth_tgt_p = 0.0
        self.smooth_tgt_r = 0.0

    def get_action(self, obs, dt):
        # 0. Observations
        px, py, altitude = obs[0], obs[1], obs[17]
        vx, vy, vz = obs[7], obs[8], obs[9] 
        wx, wy, wz = obs[10], obs[11], obs[12]
        fuel = obs[18]
        feet_contacts = obs[13:17]
        quat = obs[3:7]
        rpy = p.getEulerFromQuaternion(quat) # roll, pitch, yaw
        
        dist_p = math.sqrt(px**2 + py**2)

        # 1. Heading-to-Pad Alignment (Yaw)
        yaw_err = 0.0
        # Target +X axis towards pad
        target_yaw = math.atan2(-py, -px) if dist_p > 1.0 else rpy[2]
        err_y = target_yaw - rpy[2]
        while err_y > math.pi: err_y -= 2*math.pi
        while err_y < -math.pi: err_y += 2*math.pi
        self.yaw_pid.target = rpy[2] + err_y
        
        # Yaw PID tuning: stiffer for alignment
        self.yaw_pid.kp = 6.0
        self.yaw_pid.kd = 4.0
        yaw_out = self.yaw_pid.compute(rpy[2], dt)

        # 2. Mission State Machine
        if altitude > 50.0:
            state = "RECOV"; h_v_lim = 8.0; tilt_lim = 0.35; v_tgt = -6.0
        elif altitude > 15.0:
            state = "APPRO"; h_v_lim = 4.0; tilt_lim = 0.20; v_tgt = -3.0
        elif altitude > 5.0:
            state = "PRECI"; h_v_lim = 1.0; tilt_lim = 0.10; v_tgt = -1.5
        else:
            state = "FINAL"; h_v_lim = 0.5; tilt_lim = 0.05; v_tgt = -0.7

        # 3. Body-Frame Transformation
        cos_y = math.cos(rpy[2]); sin_y = math.sin(rpy[2])
        dx_w, dy_w = -px, -py 
        dx_l = dx_w * cos_y + dy_w * sin_y 
        dy_l = -dx_w * sin_y + dy_w * cos_y 
        vx_l = vx * cos_y + vy * sin_y
        vy_l = -vx * sin_y + vy * cos_y

        # 4. Cascaded Local Control
        # Position -> Velocity
        vx_tgt = np.clip(0.20 * dx_l, -h_v_lim, h_v_lim)
        vy_tgt = np.clip(0.20 * dy_l, -h_v_lim, h_v_lim)
        
        # Velocity -> Angle (Verified Signs)
        # Note: Nose-UP (+Pitch) produces Forward Accel.
        p_tgt = np.clip(0.4 * (vx_tgt - vx_l), -tilt_lim, tilt_lim)
        # Note: Tilt-Left (-Roll) produces Left Accel.
        r_tgt = np.clip(-0.4 * (vy_tgt - vy_l), -tilt_lim, tilt_lim)
        
        # Smoothing
        alpha_s = 0.2
        self.smooth_tgt_p = (1-alpha_s) * self.smooth_tgt_p + alpha_s * p_tgt
        self.smooth_tgt_r = (1-alpha_s) * self.smooth_tgt_r + alpha_s * r_tgt
        
        # Local Rate Targets
        wy_tgt = (self.smooth_tgt_p - rpy[1]) * 2.5
        wx_tgt = (self.smooth_tgt_r - rpy[0]) * 2.5
        
        # Rate -> Gimbal (Verified Sign: +Gimbal produces -Rate)
        gimbal_p = np.clip((wy_tgt - wy) * -0.7, -1.0, 1.0)
        gimbal_r = np.clip((wx_tgt - wx) * -0.7, -1.0, 1.0)
        
        # 5. Vertical Control
        curr_m = self.MIN_MASS + (self.FUEL_MASS * fuel)
        hover_thr = (curr_m * 9.81) / 60000.0
        self.vz_pid.target = v_tgt
        throttle_cmd = np.clip(hover_thr + self.vz_pid.compute(vz, dt) * 0.18, 0.0, 1.0)
        if altitude < 5.0 and sum(feet_contacts) >= 2 and abs(vz) < 1.0:
            throttle_cmd = 0.0
            
        # 6. Action Assembly
        action = np.zeros(16)
        action[0] = 2.0 * throttle_cmd - 1.0
        action[1] = gimbal_p
        action[2] = gimbal_r
        action[3:16] = -1.0
        
        # RCS Yaw Only (Fixed Trigger Logic)
        if yaw_out > 0.2: action[3+8]=1.0; action[3+9]=1.0 # CCW
        elif yaw_out < -0.2: action[3+10]=1.0; action[3+11]=1.0 # CW

        if altitude < 30.0: action[15] = 1.0
        
        self.last_telem = f"[{state}] Alt:{altitude:4.1f} D:{dist_p:4.1f} Thr:{throttle_cmd:4.2f} YawE:{math.degrees(err_y):4.0f}\u00b0"
        return action

from scenario_landing import setup_landing_scenario

def run_landing_mission(no_render=False, difficulty="easy"):
    print(f"\nStarting Advanced Heading-to-Pad Landing Mission [{difficulty.upper()}] (Render: {not no_render})...")
    
    render_mode = "human" if not no_render else None
    env = gym.make("RocketLander-v0", render_mode=render_mode, normalize_obs=False)
    obs = setup_landing_scenario(env, difficulty=difficulty)
    
    controller = RocketPIDController()
    controller.alt_pid.target = obs[17]
    dt = 1.0 / 30.0
    
    try:
        for i in range(5000):
            action = controller.get_action(obs, dt)
            obs, reward, terminated, truncated, info = env.step(action)
            if not no_render: env.render()
            
            if i % 10 == 0:
                print(f"T={i*dt:5.2f}s | {controller.last_telem} Pos:[{obs[0]:4.1f},{obs[1]:4.1f}] Vel:[{obs[7]:4.1f},{obs[8]:4.1f},{obs[9]:4.1f}] Contact:{int(sum(obs[13:17]))} Fuel:{obs[18]*100:4.1f}%")
            
            if terminated or truncated:
                print("\nMission Ended.")
                if reward > 500: print("SUCCESSFUL LANDING!")
                else: print("CRASH OR OUT OF BOUNDS.")
                break
    finally:
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced Rocket Lander PID Controller")
    parser.add_argument("--no-render", action="store_true", help="Disable GUI rendering")
    parser.add_argument("--difficulty", type=str, default="easy", choices=["easy", "medium", "hard", "super_hard"], help="Scenario difficulty")
    args = parser.parse_args()
    run_landing_mission(no_render=args.no_render, difficulty=args.difficulty)
