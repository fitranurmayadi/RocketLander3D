"""
Learning Lesson 5: XYZ Euler Inverse Kinematics

This script demonstrates how to solve the geometric cross-coupling and 
asymmetry of the XYZ Euler rotation sequence using inverse kinematics.

Concepts covered:
1. Thrust Vector Projection Asymmetry: Why X and Y accelerations are mathematically asymmetric.
2. Inverse Kinematics (IK) Equations: Reversing the projection to calculate correct pitch/roll.
3. Perfect Trajectory Symmetry: Achieving equal diagonal movement.
"""

import gymnasium as gym
import pybullet as p
import numpy as np
import math
import time

try:
    import rocket_lander
except ImportError:
    pass

class SimplePID:
    def __init__(self, kp, ki, kd, dt, limit=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.limit = limit
        self.integral = 0.0
        self.prev_error = 0.0
        
    def update(self, error) -> float:
        p_term = self.kp * error
        self.integral += error * self.dt
        self.integral = max(-5.0, min(5.0, self.integral))
        i_term = self.ki * self.integral
        d_term = self.kd * (error - self.prev_error) / self.dt
        self.prev_error = error
        output = p_term + i_term + d_term
        return max(-self.limit, min(self.limit, output))

class CoupledHorizontalController:
    """Calculates physically correct pitch and roll using XYZ Euler inverse kinematics."""
    def __init__(self, dt):
        self.kp = 0.4
        self.kd = 2.0
        
    def compute(self, pos_curr, pos_target, vel_curr, vel_target, current_yaw):
        pos_error = pos_target - pos_curr
        vel_error = vel_target - vel_curr
        
        # 1. Desired acceleration in world frame
        ax_world = self.kp * pos_error[0] + self.kd * vel_error[0]
        ay_world = self.kp * pos_error[1] + self.kd * vel_error[1]
        
        # 2. Transform world acceleration to yaw-aligned frame
        cy = math.cos(current_yaw)
        sy = math.sin(current_yaw)
        bx = ax_world * cy + ay_world * sy
        by = -ax_world * sy + ay_world * cy
        
        # 3. Calculate unit thrust vector components [ux, uy, uz]
        # vertical gravity (9.81) represents the baseline vertical acceleration
        tot_acc = math.sqrt(bx**2 + by**2 + 9.81**2)
        ux = bx / tot_acc
        uy = by / tot_acc
        uz = 9.81 / tot_acc
        
        # 4. XYZ Euler Inverse Kinematics
        # projects: Thrust_x = T * sin(pitch) * cos(roll)
        #           Thrust_y = -T * sin(roll)
        # So we solve:
        # roll = arcsin(-uy)
        # pitch = atan2(ux, uz)
        uy_clamped = max(-0.84, min(0.84, uy)) # clamp to prevent arcsin domain error
        desired_roll = math.asin(-uy_clamped)
        desired_pitch = math.atan2(ux, uz)
        
        # Clamp tilt to max 25 degrees (0.43 rad)
        max_tilt = 0.43
        desired_pitch = max(-max_tilt, min(max_tilt, desired_pitch))
        desired_roll = max(-max_tilt, min(max_tilt, desired_roll))
        
        return desired_pitch, desired_roll

class AttitudeController:
    def __init__(self, dt):
        self.roll_pid = SimplePID(kp=8.0, ki=0.5, kd=4.0, dt=dt, limit=1.0)
        self.pitch_pid = SimplePID(kp=8.0, ki=0.5, kd=4.0, dt=dt, limit=1.0)
        self.yaw_pid = SimplePID(kp=6.0, ki=0.5, kd=3.0, dt=dt, limit=1.0)
        
    def compute(self, roll_err, pitch_err, yaw_err) -> (float, float, float):
        u_roll = self.roll_pid.update(roll_err)
        u_pitch = self.pitch_pid.update(pitch_err)
        u_yaw = self.yaw_pid.update(yaw_err)
        return u_roll, u_pitch, u_yaw

class ActuatorMixer:
    def mix(self, throttle, u_roll, u_pitch, u_yaw, legs_deploy=False):
        action = np.zeros(16)
        action[0] = throttle * 2.0 - 1.0
        if u_roll > 0.05:
            action[3] = action[6] = abs(u_roll)
        elif u_roll < -0.05:
            action[4] = action[5] = abs(u_roll)
            
        if u_pitch > 0.05:
            action[7] = action[10] = abs(u_pitch)
        elif u_pitch < -0.05:
            action[8] = action[9] = abs(u_pitch)
            
        if u_yaw > 0.05:
            action[11] = action[14] = abs(u_yaw)
        elif u_yaw < -0.05:
            action[12] = action[13] = abs(u_yaw)
        action[15] = 1.0 if legs_deploy else -1.0
        return action

def main():
    print("==================================================")
    print("Lesson 5: XYZ Euler Inverse Kinematics")
    print("==================================================")
    print("This script will guide the rocket to climb diagonally")
    print("to target position X=20.0m, Y=20.0m, Z=50.0m.")
    print("Observe how the commanded Roll and Pitch targets are asymmetric")
    print("but the resulting coordinate movement (X, Y) is perfectly symmetric!")
    print("==================================================")
    
    env = gym.make("RocketLander-v0", render_mode="human")
    obs, _ = env.reset()
    
    rocket_id = env.unwrapped.rocketId
    p.resetBasePositionAndOrientation(rocket_id, [0.0, 0.0, 3.0], [0.0, 0.0, 0.0, 1.0])
    p.resetBaseVelocity(rocket_id, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    obs, _, _, _, _ = env.step(np.zeros(16))
    
    target_pos = np.array([20.0, 20.0, 50.0])
    target_yaw = 0.0
    
    dt = 1.0 / 60.0
    
    # Altitude PID
    alt_pid = SimplePID(kp=1.5, ki=0.0, kd=0.0, dt=dt, limit=20.0)
    vz_pid = SimplePID(kp=0.8, ki=0.15, kd=0.0, dt=dt, limit=1.0)
    gravity_ff = 0.0981
    
    # Horizontal & Attitude controllers
    hor_ctrl = CoupledHorizontalController(dt)
    att_ctrl = AttitudeController(dt)
    mixer = ActuatorMixer()
    
    try:
        for step in range(1200): # Run for 20 seconds
            # 1. Parse state
            pos_curr = obs[0:3]
            orn_quat = obs[3:7]
            euler = p.getEulerFromQuaternion(orn_quat)
            vel_curr = obs[7:10]
            
            # 2. Outer Loop: Calculate desired horizontal tilt using IK
            des_pitch, des_roll = hor_ctrl.compute(
                pos_curr=pos_curr[0:2],
                pos_target=target_pos[0:2],
                vel_curr=vel_curr[0:2],
                vel_target=np.array([0.0, 0.0]),
                current_yaw=euler[2]
            )
            
            # 3. Outer Loop: Calculate desired vz
            alt_error = target_pos[2] - pos_curr[2]
            desired_vz = alt_pid.update(alt_error)
            vz_error = desired_vz - vel_curr[2]
            throttle_adj = vz_pid.update(vz_error)
            
            tilt_cos = math.cos(euler[0]) * math.cos(euler[1])
            throttle = (gravity_ff + throttle_adj) / max(0.2, tilt_cos)
            throttle = max(0.0, min(1.0, throttle))
            
            # 4. Inner Loop: Compute RCS inputs
            roll_err = des_roll - euler[0]
            pitch_err = des_pitch - euler[1]
            yaw_err = (target_yaw - euler[2]) * math.cos(euler[1])
            u_roll, u_pitch, u_yaw = att_ctrl.compute(roll_err, pitch_err, yaw_err)
            
            # 5. Mixer
            action = mixer.mix(throttle, u_roll, u_pitch, u_yaw, legs_deploy=False)
            obs, reward, terminated, truncated, info = env.step(action)
            
            if step % 20 == 0:
                print(f"Time: {step*dt:.1f}s | Pos: ({pos_curr[0]:.1f}, {pos_curr[1]:.1f}) | Des Roll/Pitch: ({math.degrees(des_roll):.2f}, {math.degrees(des_pitch):.2f})")
                
            time.sleep(dt)
            
    except KeyboardInterrupt:
        print("\nSimulation ended by user.")
    finally:
        env.close()

if __name__ == "__main__":
    main()
