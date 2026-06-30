"""
Learning Lesson 4: RCS Attitude Control & Euler Scaling

This script implements an inner attitude control loop using the Reaction Control 
System (RCS) gas thrusters, replacing the "cheating" external torque from previous lessons.

Concepts covered:
1. RCS Torques: How firing pair combinations of cold gas thrusters yields roll, pitch, and yaw moments.
2. Euler Angular Singularities: The coupling effect where yaw rate is proportional to Wz / cos(pitch).
3. Hybrid Attitude Error: Scaling the Yaw error by cos(pitch) to prevent gimbal lock and control loop instability at high tilts.
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

class AttitudeController:
    """Computes roll, pitch, and yaw torque commands using PID loops."""
    def __init__(self, dt):
        # High gains are needed because RCS thruster force is small compared to inertia
        self.roll_pid = SimplePID(kp=8.0, ki=0.5, kd=4.0, dt=dt, limit=1.0)
        self.pitch_pid = SimplePID(kp=8.0, ki=0.5, kd=4.0, dt=dt, limit=1.0)
        self.yaw_pid = SimplePID(kp=6.0, ki=0.5, kd=3.0, dt=dt, limit=1.0)
        
    def compute(self, roll_err, pitch_err, yaw_err) -> (float, float, float):
        u_roll = self.roll_pid.update(roll_err)
        u_pitch = self.pitch_pid.update(pitch_err)
        u_yaw = self.yaw_pid.update(yaw_err)
        return u_roll, u_pitch, u_yaw

class ActuatorMixer:
    """Maps continuous control commands to individual RCS thrusters."""
    def mix(self, throttle, u_roll, u_pitch, u_yaw, legs_deploy=False):
        action = np.zeros(16)
        
        # 1. Main Engine Throttle
        action[0] = throttle * 2.0 - 1.0
        
        # 2. RCS Thrusters (12 thrusters, indices 3 to 14)
        # We fire opposing pairs to generate pure moments:
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
            
        # 3. Landing Legs
        action[15] = 1.0 if legs_deploy else -1.0
        
        return action

def main():
    print("==================================================")
    print("Lesson 4: RCS Attitude Control & Euler Scaling")
    print("==================================================")
    print("This script will guide the rocket to climb to Z=50.0m")
    print("and maintain a desired roll, pitch, and yaw target using RCS.")
    print("==================================================")
    
    env = gym.make("RocketLander-v0", render_mode="human")
    obs, _ = env.reset()
    
    rocket_id = env.unwrapped.rocketId
    p.resetBasePositionAndOrientation(rocket_id, [0.0, 0.0, 3.0], [0.0, 0.0, 0.0, 1.0])
    p.resetBaseVelocity(rocket_id, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    obs, _, _, _, _ = env.step(np.zeros(16))
    
    # Control targets
    target_pos = np.array([0.0, 0.0, 50.0])
    target_yaw = math.radians(0.0) # lock yaw at 0.0
    
    # We will test commanding a non-zero roll/pitch target (tilt by 15 deg roll and -15 deg pitch)
    # to demonstrate that our RCS controller can track it.
    target_roll = math.radians(15.0)
    target_pitch = math.radians(-15.0)
    
    dt = 1.0 / 60.0
    
    # Altitude Loops (from Lesson 2)
    alt_pid = SimplePID(kp=1.5, ki=0.0, kd=0.0, dt=dt, limit=20.0)
    vz_pid = SimplePID(kp=0.8, ki=0.15, kd=0.0, dt=dt, limit=1.0)
    gravity_ff = 0.0981
    
    # RCS Controllers
    att_ctrl = AttitudeController(dt)
    mixer = ActuatorMixer()
    
    try:
        for step in range(1200): # Run for 20 seconds
            # 1. Parse current state
            z_curr = obs[2]
            orn_quat = obs[3:7]
            euler = p.getEulerFromQuaternion(orn_quat)
            vz_curr = obs[9]
            
            # 2. Calculate attitude errors
            roll_err = target_roll - euler[0]
            pitch_err = target_pitch - euler[1]
            
            # Hybrid scaling for Yaw:
            # Under Euler angles, the yaw rate Wz is amplified by 1/cos(pitch).
            # To cancel this out, we scale down the yaw error by cos(pitch) when pitch is large,
            # keeping the yaw control loop perfectly stable at high pitch tilts.
            yaw_err = (target_yaw - euler[2]) * math.cos(euler[1])
            
            # 3. Inner Loop: Compute RCS roll, pitch, yaw inputs
            u_roll, u_pitch, u_yaw = att_ctrl.compute(roll_err, pitch_err, yaw_err)
            
            # 4. Altitude controller
            alt_error = target_pos[2] - z_curr
            desired_vz = alt_pid.update(alt_error)
            vz_error = desired_vz - vz_curr
            throttle_adj = vz_pid.update(vz_error)
            
            tilt_cos = math.cos(euler[0]) * math.cos(euler[1])
            throttle = (gravity_ff + throttle_adj) / max(0.2, tilt_cos)
            throttle = max(0.0, min(1.0, throttle))
            
            # 5. Actuator Mixing
            action = mixer.mix(throttle, u_roll, u_pitch, u_yaw, legs_deploy=False)
            
            # Step simulation
            obs, reward, terminated, truncated, info = env.step(action)
            
            if step % 20 == 0:
                print(f"Time: {step*dt:.1f}s | Roll: {math.degrees(euler[0]):.1f} deg (err={math.degrees(roll_err):.1f}) | Pitch: {math.degrees(euler[1]):.1f} deg (err={math.degrees(pitch_err):.1f})")
                
            time.sleep(dt)
            
    except KeyboardInterrupt:
        print("\nSimulation ended by user.")
    finally:
        env.close()

if __name__ == "__main__":
    main()
