import gymnasium as gym
import sys
import os
import numpy as np
import time
import pybullet as p

# Path fix for package import
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import rocket_lander 

class PID:
    def __init__(self, Kp, Ki, Kd, output_limits=(None, None)):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.output_limits = output_limits
        self.prev_error = 0.0
        self.integral = 0.0
        
    def compute(self, measurement, setpoint, dt):
        error = setpoint - measurement
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        
        # Apply limits
        low, high = self.output_limits
        if low is not None: output = max(low, output)
        if high is not None: output = min(high, output)
        
        self.prev_error = error
        return output

def main():
    print("🚀 Initializing Rocket Lander PID Autopilot...")
    env = gym.make("RocketLander-v0", render_mode="human")
    obs, info = env.reset()
    
    # --- CONTROLLER SETUP ---
    # 1. Height/Velocity (Z -> Throttle)
    # Lower gains to prevent yo-yo effect
    pid_vz = PID(Kp=0.1, Ki=0.001, Kd=0.02, output_limits=(-1.0, 1.0))
    
    # 2. Position (XY -> Target Tilt)
    # Low target tilt to keep it stable
    pid_pos_x = PID(Kp=0.05, Ki=0.0, Kd=0.1, output_limits=(-0.15, 0.15))
    pid_pos_y = PID(Kp=0.05, Ki=0.0, Kd=0.1, output_limits=(-0.15, 0.15))
    
    # 3. Attitude (Pitch/Roll Angle -> Gimbal Cmd)
    pid_pitch = PID(Kp=3.0, Ki=0.05, Kd=0.5, output_limits=(-1.0, 1.0))
    pid_roll = PID(Kp=3.0, Ki=0.05, Kd=0.5, output_limits=(-1.0, 1.0))
    
    dt = 1.0/30.0
    state = "APPROACH" 
    
    i = 0
    try:
        while True:
            # --- PARSE OBSERVATION ---
            # Obs: [x,y,z, qx,qy,qz,qw, vx,vy,vz, wx,wy,wz, legs, contact]
            pos = obs[0:3]
            quat = obs[3:7]
            vel = obs[7:10]
            ang_vel = obs[10:13]
            
            x, y, z = pos
            vx, vy, vz = vel
            roll, pitch, yaw = p.getEulerFromQuaternion(quat)
            
            # --- MISSION LOGIC (State Machine) ---
            target_vz = -2.0 # Default
            target_x, target_y = 0.0, 0.0
            legs_cmd = -1.0 # Retracted
            
            dist_xy = np.sqrt(x**2 + y**2)
            
            if state == "APPROACH":
                target_vz = 0.0 # Hover
                if z > 50: target_vz = -2.0 # Drop to hover height
                elif z < 45: target_vz = 1.0  # Climb to hover height
                
                if dist_xy < 1.0 and abs(vz) < 0.5:
                    state = "DESCENT"
                    print("🎯 Position aligned. Starting descent...")
            
            elif state == "DESCENT":
                target_vz = -2.0
                if z < 15:
                    legs_cmd = 1.0 # Deploy
                    state = "TOUCHDOWN"
                    print("🏗️ Landing legs deployed. Final approach...")
            
            elif state == "TOUCHDOWN":
                target_vz = -0.5 # Slow descent
                legs_cmd = 1.0
                
            # --- CONTROL CALCULATION ---
            
            # 1. Vertical -> Throttle
            throttle_cmd = pid_vz.compute(vz, target_vz, dt)
            
            # 2. Position -> Target Orientation
            # Correcting mapping based on physics:
            # If x=10 (too far right), error=-10, output is Negative.
            # To move left (-X), we need Nose UP (+Pitch angle).
            # So target_pitch_angle = -output.
            raw_x_output = pid_pos_x.compute(x, target_x, dt)
            raw_y_output = pid_pos_y.compute(y, target_y, dt)
            
            target_pitch_angle = -raw_x_output 
            target_roll_angle = raw_y_output  # Roll Right (+Angle) moves toward +Y.
            
            # 3. Attitude -> Gimbal
            # measurement = pitch, setpoint = target_pitch_angle.
            # If pitch=0, target=0.15, gimbal needs to increase pitch angle.
            # In env: gimbal_pitch 1.0 -> target_pitch -0.35? 
            # WAIT. If gimbal_pitch 1.0 is "Nose Up" (as verified by user), 
            # and Nose Up is positive angle (+0.35 rad), then the env has 
            # target_pitch = gimbal_pitch * 0.35. 
            # Let's check env again.
            gimbal_pitch = pid_pitch.compute(pitch, target_pitch_angle, dt)
            gimbal_roll = pid_roll.compute(roll, target_roll_angle, dt)   
            
            # --- EXECUTE ACTION ---
            # Action: [Throttle, G_Pitch, G_Roll, RCS_P, RCS_Y, RCS_R, Legs]
            action = np.zeros(7)
            action[0] = throttle_cmd
            action[1] = gimbal_pitch
            action[2] = gimbal_roll
            action[3] = 0 # No RCS during landing
            action[4] = 0
            action[5] = 0
            action[6] = legs_cmd
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            if i % 30 == 0:
                print(f"[{state}] Alt:{z:4.1f}m | Dist:{dist_xy:4.1f}m | Vz:{vz:4.2f}m/s | Thr:{throttle_cmd:4.2f}")
            
            i += 1
            if terminated:
                print("🏁 Mission Ended.")
                time.sleep(2)
                obs, info = env.reset()
                state = "APPROACH"
            
            # time.sleep(dt) # Env step already handles some timing if needed, or manual wait
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        env.close()

if __name__ == "__main__":
    main()
