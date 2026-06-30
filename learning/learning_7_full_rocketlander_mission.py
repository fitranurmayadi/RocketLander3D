"""
Learning Lesson 7: The Complete RocketLander Mission

This script combines all previous lessons into a single, complete mission simulation
managed by a Finite State Machine (FSM).

FSM Phase Transitions:
1. PRELAUNCH (t < 2.0s): Main engine off, lock orientation upright.
2. ASCENT (t < 32.0s): Climb vertically to Z=100m, then tilt forward to begin climbing to Z=1000m.
3. WAYPOINT_NAV (t < 37.0s): Fly through target waypoint coordinate at [500, 500, 1000].
4. BOOSTBACK (t < 62.0s): Tilt aggressively back (up to 57.3 deg) to cancel forward speed and fly back to Z=113m.
5. LANDING_BURN (t >= 62.0s): Keep rocket upright (max 11.5 deg tilt) for a monotonic, powered vertical descent onto the pad.
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

class AltitudeController:
    """Controls vertical throttle based on altitude and vz error."""
    def __init__(self, dt):
        self.gravity_ff = 0.0981 # TWR compensation
        self.alt_pid = SimplePID(kp=1.5, ki=0.0, kd=0.0, dt=dt, limit=20.0)
        self.vz_pid = SimplePID(kp=0.8, ki=0.15, kd=0.0, dt=dt, limit=1.0)
        
    def compute(self, alt_error, vz_error, desired_acc_z=0.0, current_alt=None) -> float:
        desired_vz = self.alt_pid.update(alt_error)
        
        # Soft descent rate limit
        if desired_vz < 0 and current_alt is not None:
            max_desc = 40.0 if current_alt > 100.0 else (15.0 if current_alt > 30.0 else 5.0)
            desired_vz = max(-max_desc, desired_vz)
            
        vz_error_total = vz_error + desired_vz
        throttle_adj = self.vz_pid.update(vz_error_total)
        
        # Add feedforward vertical acceleration (az_ff) scaled appropriately
        throttle = self.gravity_ff + (desired_acc_z * 0.05) + throttle_adj
        return max(0.0, min(1.0, throttle))

class CoupledHorizontalController:
    """Calculates desired pitch and roll using XYZ Euler inverse kinematics."""
    def __init__(self, dt):
        self.dt = dt
        
    def compute(self, pos_error, vel_error, desired_acc, current_yaw, phase) -> (float, float):
        # Dynamically adjust horizontal gains per phase to improve final touchdown accuracy
        if phase == "LandingBurn":
            kp, kd = 0.8, 2.5
        else:
            kp, kd = 0.4, 2.0
            
        ax_world = desired_acc[0] + kp * pos_error[0] + kd * vel_error[0]
        ay_world = desired_acc[1] + kp * pos_error[1] + kd * vel_error[1]
        
        cy = math.cos(current_yaw)
        sy = math.sin(current_yaw)
        bx = ax_world * cy + ay_world * sy
        by = -ax_world * sy + ay_world * cy
        
        # XYZ Euler inverse kinematics
        tot_acc = math.sqrt(bx**2 + by**2 + 9.81**2)
        ux = bx / tot_acc
        uy = by / tot_acc
        uz = 9.81 / tot_acc
        
        uy_clamped = max(-0.84, min(0.84, uy))
        desired_roll = math.asin(-uy_clamped)
        desired_pitch = math.atan2(ux, uz)
        
        # Phase-dependent maximum tilt clamp: Keep rocket upright during entry and landing burns
        if phase in ["EntryBurn", "LandingBurn"]:
            max_tilt = 0.20 # ~11.5 degrees
        else:
            max_tilt = 1.00 # ~57.3 degrees
            
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
    print("Lesson 7: The Complete RocketLander Mission")
    print("==================================================")
    
    # We use v0 with rendering enabled
    env = gym.make("RocketLander-v0", render_mode="human")
    obs, _ = env.reset()
    
    rocket_id = env.unwrapped.rocketId
    p.resetBasePositionAndOrientation(rocket_id, [0.0, 0.0, 3.0], [0.0, 0.0, 0.0, 1.0])
    p.resetBaseVelocity(rocket_id, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    obs, _, _, _, _ = env.step(np.zeros(16))
    
    # 1. Trajectory and Guidance setup (we instantiate planner and guidance from our project code!)
    from rocketlander.trajectory.planner import TrajectoryPlanner
    from rocketlander.guidance.guidance import GuidanceModule
    from rocketlander.mission.state_machine import MissionConfig
    
    config = MissionConfig()
    config.waypoint = [500.0, 500.0, 1000.0]
    
    planner = TrajectoryPlanner(config)
    guidance = GuidanceModule(planner)
    
    # Plan full 3D trajectory from initial state
    planner.plan_full_mission([0.0, 0.0, 3.0], [0.0, 0.0, 0.0])
    
    dt = 1.0 / 60.0
    alt_ctrl = AltitudeController(dt)
    hor_ctrl = CoupledHorizontalController(dt)
    att_ctrl = AttitudeController(dt)
    mixer = ActuatorMixer()
    
    mission_time = 0.0
    ignition_delay = 2.0
    
    try:
        # Step through the environment (up to 6000 steps = 100 seconds)
        for step in range(6000):
            # Parse state variables
            pos = obs[0:3]
            orn_quat = obs[3:7]
            euler = p.getEulerFromQuaternion(orn_quat)
            vel = obs[7:10]
            ang_vel = obs[10:13]
            contacts = obs[13:17]
            
            # FSM Phase Determination based on time and distance
            t = mission_time - ignition_delay
            if t < 0.0:
                phase = "PreLaunch"
            elif t < 30.0:
                phase = "Ascent"
            elif t < 35.0:
                phase = "WaypointNav"
            elif t < 60.0:
                phase = "Boostback"
            else:
                hor_dist = math.sqrt(pos[0]**2 + pos[1]**2)
                # Landing burn transitions when close to pad and below 150m
                if pos[2] < 150.0 and hor_dist < 20.0:
                    phase = "LandingBurn"
                else:
                    phase = "EntryBurn"
                    
            # Check touchdown (terminal state check)
            if phase == "LandingBurn" and sum(contacts) > 0:
                speed = math.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2)
                tilt = max(abs(euler[0]), abs(euler[1]))
                if speed < 2.5 and tilt < math.radians(10):
                    print(f"\nTOUCHDOWN SUCCESSFUL! Landed at X={pos[0]:.2f}m, Y={pos[1]:.2f}m")
                    break
                else:
                    print(f"\nCRASHED! Impact Speed: {speed:.1f} m/s, Tilt: {math.degrees(tilt):.1f} deg")
                    break
                    
            # 2. Guidance Module: Get reference coordinates
            traj_time = max(0.0, t)
            cmd = guidance.compute(pos, vel, euler, traj_time, None)
            
            # 3. Flight Control System (FCS)
            throttle = 0.0
            des_roll = 0.0
            des_pitch = 0.0
            legs_deploy = True if pos[2] < 50.0 else False
            
            if phase == "PreLaunch":
                throttle = 0.0
                des_roll = des_pitch = 0.0
            else:
                # Vertical control
                throttle = alt_ctrl.compute(
                    alt_error=cmd.pos_error[2],
                    vz_error=cmd.vel_error[2],
                    desired_acc_z=cmd.desired_acc[2],
                    current_alt=pos[2]
                )
                
                # Horizontal control
                des_pitch, des_roll = hor_ctrl.compute(
                    pos_error=cmd.pos_error,
                    vel_error=cmd.vel_error,
                    desired_acc=cmd.desired_acc,
                    current_yaw=euler[2],
                    phase=phase
                )
                
                # Ascent & Waypoint direction clamp (prevent backward tilt)
                if phase in ["Ascent", "WaypointNav"]:
                    des_pitch = max(0.0, des_pitch)
                    des_roll = min(0.0, des_roll)
                    
            # 4. Inner loop attitude control
            roll_err = des_roll - euler[0]
            pitch_err = des_pitch - euler[1]
            yaw_err = (cmd.desired_heading - euler[2]) * math.cos(euler[1]) # Euler scaling
            
            u_roll, u_pitch, u_yaw = att_ctrl.compute(roll_err, pitch_err, yaw_err)
            
            # Compensate throttle for tilt
            tilt_cos = math.cos(euler[0]) * math.cos(euler[1])
            throttle = throttle / max(0.2, tilt_cos)
            throttle = max(0.0, min(1.0, throttle))
            
            # 5. Actuator mixer
            action = mixer.mix(throttle, u_roll, u_pitch, u_yaw, legs_deploy)
            
            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            mission_time += dt
            
            if step % 20 == 0:
                print(f"\rTime: {mission_time:.1f}s | Phase: {phase:<12} | Pos: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) | Tilt: ({math.degrees(euler[0]):.1f}, {math.degrees(euler[1]):.1f}) | Throttle: {throttle*100:.1f}%", end="")
                
            time.sleep(dt)
            
    except KeyboardInterrupt:
        print("\nSimulation ended by user.")
    finally:
        env.close()

if __name__ == "__main__":
    main()
