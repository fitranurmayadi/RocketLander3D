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


class TrajectoryPlanner:
    """Generate smooth reference trajectories to landing pad"""
    
    def __init__(self):
        self.DT = 1.0 / 30.0  # Simulation timestep
        self.g = 9.81
        
    def generate_reference_trajectory(self, current_state, altitude, N=30, dt=None):
        """
        Generate a reference trajectory from current state to landing pad.
        
        Args:
            current_state: [x, y, z, vx, vy, vz]
            altitude: current altitude (for altitude-dependent limits)
            N: number of waypoints
            dt: timestep (defaults to self.DT)
        
        Returns:
            ref_states: List of [x, y, z, vx, vy, vz] waypoints
        """
        if dt is None:
            dt = self.DT
            
        ref_states = []
        
        # Goal: Landing pad at origin
        goal_pos = np.array([0.0, 0.0, 0.5])
        
        # Altitude-dependent velocity limits
        if altitude > 100.0:
            max_vel = 15.0  # Fast approach
            max_desc_rate = 8.0
        elif altitude > 50.0:
            max_vel = 10.0  # Moderate
            max_desc_rate = 5.0
        elif altitude > 20.0:
            max_vel = 5.0   # Slow
            max_desc_rate = 3.0
        else:
            max_vel = 2.0   # Very slow final approach
            max_desc_rate = 2.0
        
        curr_pos = current_state[0:3]
        
        for i in range(N):
            # Vector to goal
            diff_pos = goal_pos - curr_pos
            dist = np.linalg.norm(diff_pos)
            
            # Desired Velocity Vector
            if dist > 0.1:
                vel_dir = diff_pos / dist
                # Proportional gain: faster when far, slower when close
                v_mag = min(max_vel, dist * 1.0)
                v_ref = vel_dir * v_mag
                
                # Cap descent rate
                v_ref[2] = max(v_ref[2], -max_desc_rate)
            else:
                v_ref = np.zeros(3)
                
            # Update position for next waypoint
            curr_pos = curr_pos + v_ref * dt
            
            # Store waypoint: [x, y, z, vx, vy, vz]
            ref_state = np.zeros(6)
            ref_state[0:3] = curr_pos
            ref_state[3:6] = v_ref
            
            ref_states.append(ref_state)
            
        return np.array(ref_states)


class RocketPIDTrajectoryController:
    """PID Controller with Trajectory Tracking"""
    
    def __init__(self, target_height=300.0):
        # Constants for mass estimation
        self.MIN_MASS = 300.0
        self.MAX_MASS = 3000.0
        self.FUEL_MASS = self.MAX_MASS - self.MIN_MASS
        
        # Trajectory Planner
        self.planner = TrajectoryPlanner()
        self.ref_trajectory = None
        self.traj_update_counter = 0
        self.TRAJ_UPDATE_INTERVAL = 15  # Update every 0.5s (15 steps at 30Hz)
        
        # 1. Vertical Control (Throttle)
        self.alt_pid = PID(kp=0.25, ki=0.01, kd=0.1, target=target_height) 
        self.vz_pid = PID(kp=0.2, ki=0.01, kd=0.05, target=-2.0) 
        
        # 2. Horizontal Position Control (X/Y -> Speed)
        # These will track trajectory waypoints
        self.px_pid = PID(kp=0.05, ki=0.0, kd=0.0, target=0.0)
        self.py_pid = PID(kp=0.05, ki=0.0, kd=0.0, target=0.0)
        
        # 3. Horizontal Velocity Control (Speed -> Angle)
        self.vx_pid = PID(kp=0.1, ki=0.0, kd=0.02, target=0.0)
        self.vy_pid = PID(kp=-0.1, ki=-0.0, kd=-0.02, target=0.0)
        
        # 4. Attitude Control (Gimbal/RCS)
        # Updated with more aggressive attitude calibration gains (attitude -> desired body rates)
        # roll/pitch: target small angles to zero; yaw: also to zero.
        self.pitch_pid = PID(kp=-3.8, ki=-0.0, kd=-2.0, target=0) 
        self.roll_pid = PID(kp=-3.8, ki=-0.0, kd=-2.0, target=0) 
        self.yaw_pid = PID(kp=2.6, ki=0.0, kd=1.8, target=0)

        
        self.max_gimbal_deg = 20.0
        self.max_motor_torque = 50000.0
        self.lever_arm = 2.8
        self.last_telem = ""
        self.landed = False
        self.smooth_tgt_p = 0.0
        self.smooth_tgt_r = 0.0

    def get_action(self, obs, dt):
        # Observation Parsing
        px, py, pz = obs[0], obs[1], obs[2]
        vx, vy, vz = obs[7], obs[8], obs[9] 
        altitude = obs[17]
        fuel = obs[18]
        feet_contacts = obs[13:17]
        quat = obs[3:7]
        rpy = p.getEulerFromQuaternion(quat)
        


        # --- 1. Trajectory Update (Every 0.5s) ---
        self.traj_update_counter += 1
        if self.traj_update_counter >= self.TRAJ_UPDATE_INTERVAL or self.ref_trajectory is None:
            current_state = np.array([px, py, pz, vx, vy, vz])
            self.ref_trajectory = self.planner.generate_reference_trajectory(
                current_state, altitude, N=30
            )
            self.traj_update_counter = 0
            
        # Get current reference waypoint (first in trajectory)
        ref_waypoint = self.ref_trajectory[0]
        ref_x, ref_y, ref_z = ref_waypoint[0:3]
        ref_vx, ref_vy, ref_vz = ref_waypoint[3:6]
        
        # --- 2. Mission State Machine (Trajectory-Aware) ---
        authority_floor = 0.2
        if altitude > 50.0:
            # PHASE 1: RECOVERY - Track trajectory velocity
            self.vx_pid.target = ref_vx
            self.vy_pid.target = ref_vy
            state_str = "RECOV"
            authority_floor = 0.3 
        elif altitude > 15.0:
            # PHASE 2: APPROACH - Track trajectory with position correction
            pos_correction_x = np.clip(self.px_pid.compute(px - ref_x, dt), -5.0, 5.0)
            pos_correction_y = np.clip(self.py_pid.compute(py - ref_y, dt), -5.0, 5.0)
            self.vx_pid.target = ref_vx + pos_correction_x
            self.vy_pid.target = ref_vy + pos_correction_y
            state_str = "APPRO"
        else:
            # PHASE 3: FINAL - Zero velocity
            self.vx_pid.target = 0.0
            self.vy_pid.target = 0.0
            state_str = "FINAL"

        # Tilt limits based on altitude
        if altitude > 50.0:
            max_tilt = 0.78  # ~45 deg (Aggressive braking for high drift)
        elif altitude > 10.0:
            max_tilt = 0.14  # 8 deg
        else:
            max_tilt = 0.04  # 2 deg
        
        # --- 3. Horizontal -> Attitude (With Clipping) ---
        # Use trajectory velocity references to generate attitude targets.
        # Map desired horizontal velocity tracking error -> attitude.
        # Use ref values (ref_vx/ref_vy) to avoid persistent tilt when the measured
        # velocity still lags behind the plan.
        vel_err_x = ref_vx - vx
        vel_err_y = ref_vy - vy
        raw_target_p = np.clip(self.vx_pid.kp * vel_err_x + self.vx_pid.kd * (0.0), -max_tilt, max_tilt)
        raw_target_r = np.clip(self.vy_pid.kp * vel_err_y + self.vy_pid.kd * (0.0), -max_tilt, max_tilt)


        
        alpha_s = 0.1
        self.smooth_tgt_p = (1-alpha_s)*self.smooth_tgt_p + alpha_s*raw_target_p
        self.smooth_tgt_r = (1-alpha_s)*self.smooth_tgt_r + alpha_s*raw_target_r
        
        self.pitch_pid.target = self.smooth_tgt_p
        self.roll_pid.target = self.smooth_tgt_r
        
        # --- 4. Vertical Control (Trajectory-Aware) ---
        # Use reference vertical velocity from trajectory
        self.vz_pid.target = ref_vz
        vz_out = self.vz_pid.compute(vz, dt)
        
        # Mass Estimation & Physics Scaling
        mass_ratio = (self.MIN_MASS + self.FUEL_MASS * fuel) / self.MAX_MASS
        current_mass = self.MIN_MASS + (self.FUEL_MASS * fuel)
        hover_throttle = (current_mass * 9.81) / 60000.0
        
        # --- 5. Authority & Scaling ---
        mass_ratio = current_mass / 3000.0
        
        # Outer Loop: Orientation -> Target Rate
        target_wx = (self.roll_pid.target - rpy[0]) * 1.0
        target_wy = (self.pitch_pid.target - rpy[1]) * 1.0
        
        # Inner Loop: Rate Gains
        kp_r = -0.5 * mass_ratio
        k_damp = -0.8 * mass_ratio
        
        wx, wy, wz = obs[10], obs[11], obs[12]
        
        err_wx = target_wx - wx
        err_wy = target_wy - wy
        
        # Authority boost
        pitch_err = abs(self.pitch_pid.target - rpy[1])
        roll_err = abs(self.roll_pid.target - rpy[0])
        authority_boost = np.clip((max(pitch_err, roll_err) - 0.1) / 0.2, 0.0, 1.0) * 0.2
        
        # Throttle Logic
        dyn_floor = 0.01 
        throttle_cmd = np.clip(hover_throttle + vz_out + authority_boost, dyn_floor, 1.0)
        throttle_action = 2.0 * throttle_cmd - 1.0
        
        # --- 6. Actuator Output ---
        max_thrust = 60000.0
        current_thrust = throttle_cmd * max_thrust
        
        gimbal_limit = 1.0
        if current_thrust > 1000:
            safe_sin = self.max_motor_torque / (current_thrust * self.lever_arm)
            if safe_sin < 1.0:
                gimbal_limit = math.asin(safe_sin) / 0.35
        
        gimbal_limit = np.clip(gimbal_limit, 0.1, 1.0)
        
        # Gimbal commands (relaxed limits for trajectory tracking)
        t_scale = 1.0 / np.sqrt(np.clip(throttle_cmd, 0.05, 1.0))
        max_gimbal_rad = 0.15  # ~8.6 degrees (relaxed from 5 deg)
# NOTE: gimbal polarity in env is such that positive command produces negative torque (see verify_gimbal_polarity.py).
        # Invert commands so controller sign matches the physical effect.
        gimbal_p = -np.clip((err_wy * kp_r + (0 - wy) * k_damp) * t_scale, -max_gimbal_rad, max_gimbal_rad)
        gimbal_r = -np.clip((err_wx * kp_r + (0 - wx) * k_damp) * t_scale, -max_gimbal_rad, max_gimbal_rad)
        yaw_out = (0 - wz) * 1.5 
        
        self.last_telem = f"[{state_str}] Alt:{altitude:.1f} RefV:[{ref_vx:.1f},{ref_vy:.1f},{ref_vz:.1f}] Thr:{throttle_cmd:.2f}"

        # Construct Action Vector
        action = np.zeros(16)
        action[0] = throttle_action
        action[3:16] = -1.0
        
        action[1] = np.clip(gimbal_p, -gimbal_limit, gimbal_limit)
        action[2] = np.clip(gimbal_r, -gimbal_limit, gimbal_limit)
        
        # --- 7. RCS Integration ---
        t = 0.15
        w_t = 0.1 
        
        # PITCH
        if gimbal_p > t or wy > w_t:
            action[3+0]=1.0; action[3+5]=1.0 
        elif gimbal_p < -t or wy < -w_t:
            action[3+1]=1.0; action[3+4]=1.0 
            
        # ROLL
        if gimbal_r > t or wx > w_t:
            action[3+2]=1.0; action[3+7]=1.0 
        elif gimbal_r < -t or wx < -w_t:
            action[3+3]=1.0; action[3+6]=1.0 
            
        # YAW
        if yaw_out > 0.1 or wz > 0.1:
            action[3+8]=1.0; action[3+9]=1.0 
        elif yaw_out < -0.1 or wz < -0.1:
            action[3+10]=1.0; action[3+11]=1.0 
            
        # Legs
        if altitude < 50.0:
            action[15] = 1.0
            
        return action


from scenario_landing import setup_landing_scenario

def run_landing_mission(no_render=False, difficulty="medium"):
    print(f"\\nStarting PID+Trajectory Landing Mission (Render: {not no_render}, Diff: {difficulty})...")
    
    render_mode = "human" if not no_render else None
    env = gym.make("RocketLander-v0", render_mode=render_mode)
    
    obs = setup_landing_scenario(env, difficulty=difficulty)
    
    controller = RocketPIDTrajectoryController()
    controller.alt_pid.target = obs[17]
    dt = 1.0 / 30.0
    
    try:
        for i in range(5000):  # 166 seconds at 30Hz (was 1000/33s)
            action = controller.get_action(obs, dt)
            obs, reward, terminated, truncated, info = env.step(action)
            
            if not no_render:
                env.render()
            
            if i % 10 == 0:
                rpy_now = p.getEulerFromQuaternion(obs[3:7])
                rpy_deg = [math.degrees(a) for a in rpy_now]
                fuel_pct = obs[18] * 100.0
                print(f"T={i/30:.2f}s | {controller.last_telem} | RPY: {rpy_deg[1]:.1f}° | Fuel: {fuel_pct:.1f}%")
            
            if terminated or truncated:
                print("\\nMission Ended.")
                if reward > 0: 
                    print("SUCCESSFUL LANDING! 🚀")
                else: 
                    print("CRASH OR OUT OF BOUNDS. 💥")
                    print(f"Final State: Alt={obs[17]:.1f}, Vel={obs[7:10]}")
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-render", action="store_true", help="Run without PyBullet GUI")
    parser.add_argument("--difficulty", type=str, default="medium", 
                       choices=["easy", "medium", "hard", "super_hard", "extreme"],
                       help="Landing scenario difficulty")
    args = parser.parse_args()
    
    run_landing_mission(no_render=args.no_render, difficulty=args.difficulty)
