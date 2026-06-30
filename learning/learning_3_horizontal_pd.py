"""
Learning Lesson 3: Horizontal Control & Positioning

This script shows how to translate horizontal coordinate errors (X, Y) 
into commanded body tilt angles (pitch, roll) to navigate the rocket in 3D.

Concepts covered:
1. Engine Thrust Projection: Acceleration in X and Y requires tilting the rocket nozzle.
2. Outer Horizontal PD Loop: Converting X/Y position and velocity error to desired pitch/roll.
3. Decoupled Frame Translation: Translating world coordinate errors to yaw-aligned body frame.
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

class HorizontalController:
    """Calculates desired pitch and roll angles to achieve target horizontal position."""
    def __init__(self, dt):
        self.kp = 0.4
        self.kd = 2.0
        
    def compute(self, pos_curr, pos_target, vel_curr, vel_target, current_yaw):
        # 1. Calculate world frame errors
        pos_error = pos_target - pos_curr
        vel_error = vel_target - vel_curr
        
        # 2. Desired world frame acceleration
        ax_world = self.kp * pos_error[0] + self.kd * vel_error[0]
        ay_world = self.kp * pos_error[1] + self.kd * vel_error[1]
        
        # 3. Rotate world frame acceleration to yaw-aligned frame (body frame)
        cy = math.cos(current_yaw)
        sy = math.sin(current_yaw)
        
        # bx, by represent acceleration requirements aligned with rocket headings
        bx = ax_world * cy + ay_world * sy
        by = -ax_world * sy + ay_world * cy
        
        # 4. Convert acceleration to tilt angle: a = g * tan(theta) => theta = atan(a / g)
        # Positive pitch tilts forward (moves +X)
        # Negative roll tilts right (moves +Y)
        desired_pitch = math.atan2(bx, 9.81)
        desired_roll = math.atan2(-by, 9.81)
        
        # Clamp tilt angle to max 25 degrees (0.43 rad) to maintain stability
        max_tilt = 0.43
        desired_pitch = max(-max_tilt, min(max_tilt, desired_pitch))
        desired_roll = max(-max_tilt, min(max_tilt, desired_roll))
        
        return desired_pitch, desired_roll

def main():
    print("==================================================")
    print("Lesson 3: Horizontal Control & Positioning")
    print("==================================================")
    print("This script will guide the rocket to climb to Z=50.0m")
    print("and then hover horizontally to target X=15.0m, Y=15.0m.")
    print("==================================================")
    
    env = gym.make("RocketLander-v0", render_mode="human")
    obs, _ = env.reset()
    
    rocket_id = env.unwrapped.rocketId
    p.resetBasePositionAndOrientation(rocket_id, [0.0, 0.0, 3.0], [0.0, 0.0, 0.0, 1.0])
    p.resetBaseVelocity(rocket_id, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    obs, _, _, _, _ = env.step(np.zeros(16))
    
    # Target Coordinates
    target_pos = np.array([15.0, 15.0, 50.0])
    
    dt = 1.0 / 60.0
    
    # Altitude Loops (from Lesson 2)
    alt_pid = SimplePID(kp=1.5, ki=0.0, kd=0.0, dt=dt, limit=20.0)
    vz_pid = SimplePID(kp=0.8, ki=0.15, kd=0.0, dt=dt, limit=1.0)
    gravity_ff = 0.0981
    
    # Horizontal Loop
    hor_ctrl = HorizontalController(dt)
    
    try:
        for step in range(1200): # Run for 20 seconds
            # 1. Parse current state
            pos_curr = obs[0:3]
            orn_quat = obs[3:7]
            euler = p.getEulerFromQuaternion(orn_quat)
            vel_curr = obs[7:10]
            
            # Current Yaw angle
            yaw = euler[2]
            
            # 2. Outer Loop: Calculate desired horizontal tilt
            des_pitch, des_roll = hor_ctrl.compute(
                pos_curr=pos_curr[0:2],
                pos_target=target_pos[0:2],
                vel_curr=vel_curr[0:2],
                vel_target=np.array([0.0, 0.0]),
                current_yaw=yaw
            )
            
            # 3. Outer Loop: Calculate desired Vz
            alt_error = target_pos[2] - pos_curr[2]
            desired_vz = alt_pid.update(alt_error)
            
            # 4. Inner Loop: Calculate vertical throttle
            vz_error = desired_vz - vel_curr[2]
            throttle_adj = vz_pid.update(vz_error)
            
            # Scale throttle to compensate for tilt (if tilted, vertical thrust is reduced)
            tilt_cos = math.cos(euler[0]) * math.cos(euler[1])
            throttle = (gravity_ff + throttle_adj) / max(0.2, tilt_cos)
            throttle = max(0.0, min(1.0, throttle))
            
            # 5. Build Gymnasium action
            action = np.zeros(16)
            action[0] = throttle * 2.0 - 1.0
            
            # For this lesson, we cheat to rotate the rocket directly to the desired pitch/roll
            # (In Lesson 4 & 5, we will replace this cheating torque with actual RCS controls!)
            k_ang = 400.0
            k_damp = 80.0
            torque_roll = k_ang * (des_roll - euler[0]) - k_damp * obs[10]
            torque_pitch = k_ang * (des_pitch - euler[1]) - k_damp * obs[11]
            torque_yaw = -k_ang * euler[2] - k_damp * obs[12]
            p.applyExternalTorque(rocket_id, -1, [torque_roll, torque_pitch, torque_yaw], p.LINK_FRAME)
            
            # Step env
            obs, reward, terminated, truncated, info = env.step(action)
            
            if step % 20 == 0:
                print(f"Time: {step*dt:.1f}s | Pos: ({pos_curr[0]:.1f}, {pos_curr[1]:.1f}, {pos_curr[2]:.1f}) | Des R/P: ({math.degrees(des_roll):.1f}, {math.degrees(des_pitch):.1f})")
                
            time.sleep(dt)
            
    except KeyboardInterrupt:
        print("\nSimulation ended by user.")
    finally:
        env.close()

if __name__ == "__main__":
    main()
