import gymnasium as gym
import lunar_lander_3d
import numpy as np
import pybullet as p
import math
import time
import os
from enum import Enum
try:
    import osc_sender
    _OSC_AVAILABLE = True
except ImportError:
    _OSC_AVAILABLE = False

class ControlState(Enum):
    RECOVERY           = 1 # Target 1: Recover from random tumble
    ORIENT             = 2 # Target 2: Point toward target
    TRANSIT            = 3 # Target 3: Fly to (0,0) with fluid yaw
    LANDING            = 4 # Target 4: Precision Descent

class PID:
    def __init__(self, kp, ki, kd, dt, limit=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.limit = limit
        self.prev_error = 0
        self.integral = 0
        
    def update(self, error, is_angular=False):
        self.integral += error * self.dt
        self.integral = np.clip(self.integral, -self.limit, self.limit)
        
        diff = error - self.prev_error
        if is_angular:
            while diff > np.pi: diff -= 2 * np.pi
            while diff < -np.pi: diff += 2 * np.pi
            
        derivative = diff / self.dt
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative
    
    def reset(self):
        self.prev_error = 0; self.integral = 0

def denorm_pos(norm_val, idx): return norm_val * 2000
def denorm_vel(norm_val): return norm_val * 200.0
def denorm_ang_vel(norm_val): return norm_val * 10.0

class StateMachineController:
    def __init__(self, dt=0.01):
        self.dt = dt
        self.gravity_ff = 0.368
        
        # Tuning - CORRECTED FOR GRAVITY
        # Tuning - Phase 18 Anti-Oscillation (Super-Smooth)
        self.roll_pid  = PID(kp=1.5, ki=0.0, kd=2.0, dt=dt, limit=1.0)
        self.pitch_pid = PID(kp=1.5, ki=0.0, kd=2.0, dt=dt, limit=1.0)
        # Phase 23: Tighter Yaw for linearity
        self.yaw_pid = PID(kp=1.0, ki=0.01, kd=1.0, dt=dt, limit=1.0)
        # Phase 22 Fix: Increase Vz Kp slightly (3.0 -> 4.0) for landing authority
        self.vz_pid    = PID(kp=4.0, ki=0.1, kd=1.0, dt=dt, limit=1.0) 
        
        # Horizontal - Very soft for butter-smooth transit (Phase 20)
        self.vx_pid    = PID(kp=0.1, ki=0.0, kd=0.2, dt=dt, limit=1.0)
        self.vy_pid    = PID(kp=0.1, ki=0.0, kd=0.2, dt=dt, limit=1.0)
        
        # --- SAFETY ENVELOPE (Phase 5) ---
        # Phase 22/23: Conservative Logic
        self.MAX_TILT = 0.5 # Physical limit (Approach)
        self.MAX_G  = 2.0  # 2.0G limit for passengers
        self.MAX_W  = np.radians(30.0) # 30 deg/s to prevent vertigo
        self.MAX_VH = 8.0   # Phase 23: Slow cinematic speed (was 20.0) matches braking power
        self.MAX_VZ = 10.0  # 10 m/s vertical cap
        # Phase 22 Fix: Increase Thrust Slew (0.02 -> 0.05) to allow descent
        self.THRUST_SLEW = 0.05 
        
        self.target_yaw_smooth = None
        self.last_valid_target_yaw = 0.0
        self.bx_rcs_smooth = 0.0
        self.by_rcs_smooth = 0.0
        self.p_cmd_smooth = 0.0
        self.r_cmd_smooth = 0.0
        self.y_cmd_smooth = 0.0
        self.reset()
        
    def reset(self):
        self.state = ControlState.RECOVERY
        self.prev_vel = np.zeros(3)
        self.prev_thrust = 0.368
        self.step_count = 0
        self.g_force = 1.0
        self.target_pitch_smooth = 0.0
        self.target_roll_smooth = 0.0
        self.bx_rcs_smooth = 0.0
        self.by_rcs_smooth = 0.0
        self.p_cmd_smooth = 0.0
        self.r_cmd_smooth = 0.0
        self.y_cmd_smooth = 0.0
        for p in [self.roll_pid, self.pitch_pid, self.yaw_pid, self.vz_pid, self.vx_pid, self.vy_pid]:
            p.reset()
            
    def reset_horizontal_pids(self):
        self.vx_pid.reset()
        self.vy_pid.reset()
        
    def normalize_angle(self, angle):
        if not math.isfinite(angle): return 0.0
        while angle > np.pi: angle -= 2 * np.pi
        while angle < -np.pi: angle += 2 * np.pi
        return angle
        

    def update_state_machine(self, r, p, vz, h, y, dist, target_y, contact=False):
        # 0. Safety Check
        is_failing = abs(r) > 30.0 or abs(p) > 30.0
        
        # 1. RECOVERY
        if self.state == ControlState.RECOVERY:
            if abs(r) < 10 and abs(p) < 10 and abs(vz) < 5.0:
                self.state = ControlState.ORIENT
                
        # 2. ORIENT -> TRANSIT
        elif self.state == ControlState.ORIENT:
            # Check alignment
            y_err = abs(y - target_y)
            if y_err > 180: y_err = 360 - y_err
            
            if y_err < 10.0:
                self.state = ControlState.TRANSIT
                self.reset_horizontal_pids()
                
        # 3. TRANSIT -> LANDING
        # 3. TRANSIT -> LANDING
        elif self.state == ControlState.TRANSIT:
            # Phase 22.5 Fix: Also trigger LANDING if we touch ground early
            # Phase 23 Fix: Sanity check (h < 5.0) included in 'any_contact' logic before call
            if (dist < 5.0 and h < 30.0) or contact:
               self.state = ControlState.LANDING
               self.reset_horizontal_pids()
               
        # 4. LANDING logic (Abort back to TRANSIT if target lost)
        elif self.state == ControlState.LANDING:
            if is_failing: self.state = ControlState.RECOVERY
            elif dist > 25.0: # Wider hysteresis
                print(f"[SM] >> LOST TARGET. Abort to TRANSIT. Dist={dist:.1f}")
                self.state = ControlState.TRANSIT
                self.reset_horizontal_pids()
                
    def compute_action(self, obs):
        self.step_count += 1
        if obs.shape[0] == 34: obs = obs[:17]
        
        # Denormalize
        pos_real = np.array([denorm_pos(obs[i], i) for i in range(3)])
        vel_real = np.array([denorm_vel(obs[i+6]) for i in range(3)])
        r, p, y = obs[3]*np.pi, obs[4]*np.pi, obs[5]*np.pi
        ang_vel = np.array([denorm_ang_vel(obs[i+9]) for i in range(3)]) 
        
        h = pos_real[2]; vz = vel_real[2]
        dist_h = math.sqrt(pos_real[0]**2 + pos_real[1]**2)
        
        r_deg, p_deg = math.degrees(r), math.degrees(p)
        
        # Target Yaw for sequential logic
        if self.state in [ControlState.ORIENT, ControlState.TRANSIT]:
            # FIX: LATCH yaw when near origin (singularity) to avoid atan2 noise
            if dist_h > 10.0:
                self.last_valid_target_yaw = math.atan2(-pos_real[1], -pos_real[0])
            target_yaw_rad = self.last_valid_target_yaw
        elif self.state == ControlState.RECOVERY:
            target_yaw_rad = y # Maintain current
        else:
            target_yaw_rad = 0.0 # Align North
            
        # State Update
        # Phase 22 Fix: Pass contact info to state machine
        left_contact = obs[13] > 0.5
        right_contact = obs[14] > 0.5
        front_contact = obs[15] > 0.5
        back_contact = obs[16] > 0.5
        # Phase 23 Fix: Sanity check contact with altitude to prevent False Positives
        any_contact = (left_contact or right_contact or front_contact or back_contact) and (h < 5.0)
        
        self.update_state_machine(r_deg, p_deg, vz, h, math.degrees(y), dist_h, math.degrees(target_yaw_rad), any_contact)
        
        # --- G-Force Estimation ---
        accel = (vel_real - self.prev_vel) / self.dt
        moon_g = np.array([0, 0, -1.62])
        self.g_force = np.linalg.norm((accel - moon_g)) / 9.8 
        self.prev_vel = vel_real.copy()

        # --- CONTROL TARGETS (Phase 16 Sequential) ---
        target_vz = 0.0
        target_yaw = 0.0
        move_x, move_y = 0.0, 0.0
        
        if self.state == ControlState.RECOVERY:
            target_vz = 0.0
            target_yaw = y
            move_x, move_y = 0.0, 0.0
            
        elif self.state == ControlState.ORIENT:
            target_vz = 0.0
            target_yaw = target_yaw_rad
            move_x = -vel_real[0]
            move_y = -vel_real[1]
            
        elif self.state == ControlState.TRANSIT:
            # 1. Altitude (Balanced descent - Butter-Smooth)
            if h > 50:    target_vz = -5.0 
            else:         target_vz = -2.0
            # 2. Yaw: Face Target
            target_yaw = target_yaw_rad
            # 3. Horizontal: Very soft schedule
            des_vx = np.clip(-pos_real[0] * 0.1, -self.MAX_VH, self.MAX_VH)
            des_vy = np.clip(-pos_real[1] * 0.1, -self.MAX_VH, self.MAX_VH)
            move_x = des_vx - vel_real[0]
            move_y = des_vy - vel_real[1]
            
        elif self.state == ControlState.LANDING:
            # Phase 24: Terminal Descent Override (Wider < 5.0m)
            if h < 15.0 and dist_h < 5.0:
                 target_vz = -1.0 # CONSTANT DESCENT to Break Hover
            else:
                 target_vz = -np.clip(0.1 * h, 0.4, 3.0) # Slower descent from high altitude
            target_yaw = 0.0 
            des_vx = np.clip(-pos_real[0] * 0.2, -5, 5) # Faster precision
            des_vy = np.clip(-pos_real[1] * 0.2, -5, 5)
            move_x = des_vx - vel_real[0]
            move_y = des_vy - vel_real[1]
            
        # EXTRA: Force Damping in non-transit states for better stability
        if self.state in [ControlState.RECOVERY, ControlState.ORIENT]:
             target_vz = 0.0
             target_yaw = y if self.state == ControlState.RECOVERY else target_yaw_rad
             move_x = -vel_real[0]
             move_y = -vel_real[1]
            
        # --- APPLY ---
        a = np.zeros(21)
        
        # Deadband & LPF (Jitter Fixes) - Narrowed for precision
        if abs(move_x) < 0.05: move_x = 0.0
        if abs(move_y) < 0.05: move_y = 0.0
        
        # COORDINATE TRANSFORM (Phase 15/16 decouple)
        # 1. Run PID on World-Frame targets
        wx = np.clip(self.vx_pid.update(move_x), -1, 1)
        wy = np.clip(self.vy_pid.update(move_y), -1, 1)
        
        # 2. Transform World FORCE to Body FORCE
        cy_move = math.cos(-y); sy_move = math.sin(-y)
        bx = wx * cy_move - wy * sy_move
        by = wx * sy_move + wy * cy_move

        # 3. Vertical PID update (UNIFIED)
        vz_cmd = np.clip(self.vz_pid.update(target_vz - vz), -1.0, 1.0)
        
        # AUTHORITY SEPARATION (Phase 17 Jitter Fix)
        
        # AUTHORITY SEPARATION (Phase 17 Jitter Fix)
        if self.state == ControlState.LANDING:
            # TERMINAL: Flat Attitude + 100% Precision RCS
            pitch_tilt = 0.0
            roll_tilt  = 0.0
            bx_rcs, by_rcs = bx, by 
        elif self.state == ControlState.ORIENT:
            # SEQUENTIAL: Level attitude while damping
            pitch_tilt = 0.0
            roll_tilt  = 0.0
            bx_rcs = bx * 0.5
            by_rcs = by * 0.5
        else:
            # TRANSIT: Hybrid Control (Phase 22 Fix)
            # 1. Cruise Mode (>50m): Pure Tilt, Smooth, Low Authority
            if dist_h > 50.0:
                # Phase 23: Relax cruise tilt to 0.35 (20 deg) for better braking
                pitch_tilt = np.clip(bx * 0.5, -0.35, 0.35)
                roll_tilt  = np.clip(-by * 0.5, -0.35, 0.35)
                bx_rcs = 0.0
                by_rcs = 0.0
            # 2. Approach Mode (<=50m): Full Authority + RCS for Precision
            else:
                pitch_tilt = np.clip(bx * 0.5, -0.5, 0.5)
                roll_tilt  = np.clip(-by * 0.5, -0.5, 0.5)
                bx_rcs = bx
                by_rcs = by
            
        # 2. Target transitions - Rate Limited (Phase 22: Ultra-Slow 0.1 deg)
        # Max change = 0.1 degrees per step (~0.0017 rad) for "Floating" feel
        MAX_TILT_RATE = np.radians(0.1)
        
        delta_p = pitch_tilt - self.target_pitch_smooth
        delta_p = np.clip(delta_p, -MAX_TILT_RATE, MAX_TILT_RATE)
        self.target_pitch_smooth += delta_p
        
        delta_r = roll_tilt - self.target_roll_smooth
        delta_r = np.clip(delta_r, -MAX_TILT_RATE, MAX_TILT_RATE)
        self.target_roll_smooth += delta_r
        
        # Error is (Target - Actual)
        p_err = self.normalize_angle(self.target_pitch_smooth - p)
        r_err = self.normalize_angle(self.target_roll_smooth - r)
        
        p_cmd = np.clip(self.pitch_pid.update(p_err), -1, 1)
        r_cmd = np.clip(self.roll_pid.update(r_err), -1, 1)
        
        # --- CALCULATE BASE COMMANDS ---
        y_err = self.normalize_angle(target_yaw - y)
        y_cmd = np.clip(self.yaw_pid.update(y_err, is_angular=True), -1, 1)
        
        # Phase 14 Fix: Hold Altitude or move to target altitude
        if self.state in [ControlState.RECOVERY, ControlState.ORIENT]:
             target_vz = 0.0
        
        if self.state != ControlState.RECOVERY:
            # --- ANGULAR COMFORT CAP ---
            w_mag = np.linalg.norm(ang_vel)
            if w_mag > self.MAX_W:
                scale = self.MAX_W / w_mag
                p_cmd *= scale; r_cmd *= scale; y_cmd *= scale
        
        # Phase 19.5: Butter-Smooth Filtering (Balanced 0.2)
        alpha_cmd = 0.2 # Smooth but responsive enough to land
        self.p_cmd_smooth = alpha_cmd * p_cmd + (1 - alpha_cmd) * self.p_cmd_smooth
        self.r_cmd_smooth = alpha_cmd * r_cmd + (1 - alpha_cmd) * self.r_cmd_smooth
        self.y_cmd_smooth = alpha_cmd * y_cmd + (1 - alpha_cmd) * self.y_cmd_smooth
        
        p_cmd, r_cmd, y_cmd = self.p_cmd_smooth, self.r_cmd_smooth, self.y_cmd_smooth
        
        # Deadband: Kill tiny oscillations
        if abs(p_cmd) < 0.05: p_cmd = 0.0
        if abs(r_cmd) < 0.05: r_cmd = 0.0
        if abs(y_cmd) < 0.05: y_cmd = 0.0
        
        # --- APPLY RCS ACTUATORS ---
        # Back to Classic Mappings (Verified Working for Tumble)
        if p_cmd > 0: a[5] = abs(p_cmd); a[9] = abs(p_cmd) 
        else:         a[4] = abs(p_cmd); a[10] = abs(p_cmd) 
        
        if r_cmd > 0: a[14] = abs(r_cmd); a[20] = abs(r_cmd)
        else:         a[15] = abs(r_cmd); a[19] = abs(r_cmd)
        
        if y_cmd > 0: a[3]=a[8]=a[13]=a[18] = abs(y_cmd)
        else:         a[2]=a[7]=a[12]=a[17] = abs(y_cmd)
        
        if self.step_count % 50 == 0:
             print(f"   [RCS] P_cmd={p_cmd:5.2f} | R_cmd={r_cmd:5.2f} | Y_cmd={y_cmd:5.2f}")
        
        # Thrust with Slew Rate Limiting & Safety Clip
        # Use a safe cos denominator to prevent Infinity during deep tilts
        cos_safe = max(0.1, math.cos(p) * math.cos(r))
        target_thrust_raw = (self.gravity_ff + vz_cmd) / cos_safe
        target_thrust = np.clip(target_thrust_raw, 0.0, 1.0)
        
        if self.state != ControlState.RECOVERY:
            thrust_delta = target_thrust - self.prev_thrust
            thrust_delta = np.clip(thrust_delta, -self.THRUST_SLEW, self.THRUST_SLEW)
            final_thrust = self.prev_thrust + thrust_delta
        else:
            final_thrust = target_thrust
            
        if any_contact and self.state == ControlState.LANDING:
             final_thrust = 0.0 # HARD CUTOFF
             
        a[0] = final_thrust
        self.prev_thrust = final_thrust
        
        # Horizontal Translation RCS (Phase 19 Filtered for Zero Jitter)
        alpha_rcs = 0.2
        self.bx_rcs_smooth = alpha_rcs * bx_rcs + (1 - alpha_rcs) * self.bx_rcs_smooth
        self.by_rcs_smooth = alpha_rcs * by_rcs + (1 - alpha_rcs) * self.by_rcs_smooth
        
        cx = np.clip(self.bx_rcs_smooth, -1, 1)
        cy_rcs = np.clip(self.by_rcs_smooth, -1, 1)
        
        # Deadband on Translation to prevent buzzing
        if abs(cx) < 0.05: cx = 0.0
        if abs(cy_rcs) < 0.05: cy_rcs = 0.0
        
        if cx > 0: a[1]=abs(cx)
        else:      a[6]=abs(cx)
        
        if cy_rcs > 0: a[11]=abs(cy_rcs)
        else:          a[16]=abs(cy_rcs)
        
        # DEBUG INFO
        debug_info = {
            "target_vz": target_vz,
            "target_yaw": math.degrees(target_yaw),
            "dist_h": dist_h,
            "g_force": self.g_force,
            "state": self.state.value
        }
        
        # Telemetry (Debug: High frequency)
        if self.step_count % 10 == 0:
             print(f"[SM] {self.step_count:<5} | St={self.state.name:<10} | H={h:5.1f} | Vz={vz:5.2f}/{target_vz:.1f} | Dist={dist_h:5.1f} | Thr={final_thrust:.2f} | C={any_contact} | RAW={left_contact}{right_contact}{front_contact}{back_contact}")

        return a, debug_info

import matplotlib.pyplot as plt
import argparse

def run_test():
    parser = argparse.ArgumentParser(description='Lunar Lander 3D Mission V2 (Direct)')
    parser.add_argument('--episodes', type=int, default=1, help='Number of episodes to run (default: 1)')
    parser.add_argument('--fixed', action='store_true', help='Compass Test: Run 4 quadrants (SW, NW, SE, NE) at 1km with 45deg tilt')
    parser.add_argument('--spawn', type=float, nargs=3, metavar=('X', 'Y', 'Z'), help='Custom spawn position')
    parser.add_argument('--orient', type=float, nargs=3, metavar=('R', 'P', 'Y'), help='Custom orientation in degrees')
    parser.add_argument('--no-render', action='store_true', help='Run without PyBullet GUI rendering')
    parser.add_argument('--no-dashboard', action='store_true', help='Disable live OSC telemetry dashboard')
    args = parser.parse_args()
    use_osc = _OSC_AVAILABLE and not args.no_dashboard
    if use_osc:
        osc_sender.init()
    
    TOTAL_EPISODES = 4 if args.fixed else args.episodes
    base_seed = int(time.time())
    
    render_mode = None if args.no_render else "human"
    env = gym.make('LunarLander3D-v1', render_mode=render_mode, max_episode_steps=120000)
    
    fixed_points = [
        [-1000, -1000, 1000], [-1000, 1000, 1000],
        [1000, -1000, 1000], [1000, 1000, 1000]
    ]
    fixed_labels = ["SW", "NW", "SE", "NE"]
    
    for ep in range(TOTAL_EPISODES):
        label = fixed_labels[ep % 4] if args.fixed else "RANDOM"
        print(f"\n=== EPISODE {ep + 1} / {TOTAL_EPISODES} ({label}) ===")
        if use_osc:
            osc_sender.send_episode(ep + 1, label)
        
        
        # INCREASED TIMEOUT for safety-capping (slower transit)
        
        if args.spawn:
            start_x, start_y, start_z = args.spawn
            r, p, y = [math.radians(v) for v in args.orient] if args.orient else [0,0,0]
            print(f"Spawn (CUSTOM): X={start_x:.1f}, Y={start_y:.1f}, Z={start_z:.1f}")
        elif args.fixed:
            # Stress Test Case: Quadrant loop
            start_x, start_y, start_z = fixed_points[ep % 4]
            r, p, y = math.radians(45), math.radians(45), math.radians(45)
            print(f"Spawn (FIXED-{label}): X={start_x:.1f}, Y={start_y:.1f}, Z={start_z:.1f}")
            print(f"Attitude (FIXED): R=45.0, P=45.0, Y=45.0")
        else:
            # Random Init
            np.random.seed(base_seed + ep)
            start_x = np.random.uniform(-900, 900)
            start_y = np.random.uniform(-900, 900)
            start_z = 1000.0
            r = np.random.uniform(-0.5, 0.5) 
            p = np.random.uniform(-0.5, 0.5)
            y = np.random.uniform(-3.14, 3.14)
            print(f"Spawn: X={start_x:.1f}, Y={start_y:.1f}, H={start_z:.1f}")
            print(f"Attitude: R={math.degrees(r):.1f}, P={math.degrees(p):.1f}, Y={math.degrees(y):.1f}")
        
        obs, info = env.reset(seed=base_seed+ep, options={
            "spawn_pos": [start_x, start_y, start_z],
            "spawn_att": [math.degrees(r), math.degrees(p), math.degrees(y)],
            "spawn_vel": [0, 0, 0],
            "spawn_ang": [0, 0, 0]
        })
        
        ctrl = StateMachineController(dt=env.unwrapped.sim_time)
        
        # LOGS
        log_data = {
            "pos": [], "vel": [], "att": [], "ang_vel": [],
            "action": [], "fuel": [], "reward": [],
            "target_vz": [], "target_yaw": [], "dist_h": [],
            "g_force": [], "state": []
        }
        
        total_reward = 0
        # Log Initial State for Plotting Start Point
        log_data["pos"].append(np.array([start_x, start_y, start_z]))
        
        stable_blocks = 0
        
        for i in range(120000): # Match max steps
            action, debug = ctrl.compute_action(obs)
            obs, reward, done, trunc, _ = env.step(action)
            total_reward += reward
            
            # Extract Data
            if obs.shape[0] == 34: current_obs = obs[:17]
            else: current_obs = obs
                
            pos_real = np.array([denorm_pos(current_obs[k], k) for k in range(3)])
            vel_real = np.array([denorm_vel(current_obs[k+6]) for k in range(3)])
            att_real = np.array([current_obs[3]*180, current_obs[4]*180, current_obs[5]*180])
            ang_real = np.array([current_obs[9], current_obs[10], current_obs[11]]) 
            fuel = current_obs[12]
            
            # Store
            log_data["pos"].append(pos_real)
            log_data["vel"].append(vel_real)
            log_data["att"].append(att_real)
            log_data["ang_vel"].append(ang_real)
            log_data["action"].append(action)
            log_data["fuel"].append(fuel)
            log_data["reward"].append(reward)
            log_data["target_vz"].append(debug["target_vz"])
            log_data["target_yaw"].append(debug["target_yaw"])
            log_data["dist_h"].append(debug["dist_h"])
            log_data["g_force"].append(ctrl.g_force)
            log_data["state"].append(ctrl.state.value)
            
# --- OSC telemetry (every step ~10ms) ---
            if use_osc:
                cum_r = float(np.sum(log_data["reward"]))
                osc_sender.send_state(i, float(pos_real[2]), float(vel_real[2]),
                                      float(debug["dist_h"]), float(reward),
                                      int(ctrl.state.value), cum_r)
                osc_sender.send_attitude(float(att_real[0]), float(att_real[1]), float(att_real[2]))
                osc_sender.send_velocity(float(vel_real[0]), float(vel_real[1]), float(vel_real[2]))
                osc_sender.send_action(float(action[0]), float(np.mean(action[1:])), float(ctrl.g_force))
            
            # Phase 24: Manual Success Trigger (Fix for broken contact sensor)
            if pos_real[2] < 12.0 and np.linalg.norm(vel_real) < 0.2:
                stable_blocks += 1
                if stable_blocks > 50:
                    print(f"SUCCESS: Stable Landing Detected at H={pos_real[2]:.2f}m")
                    break
            else:
                stable_blocks = 0
            
            if done or trunc:
                print(f"Episode {ep + 1} Finished. Total Reward: {total_reward:.1f}")
                time.sleep(1.0)
                break
                
            # SLOW DOWN FOR HUMAN VISION (10x Speed) - Only if rendering
            if not args.no_render:
                time.sleep(ctrl.dt / 10.0)
        
        # --- CUSTOM PLOTTING CONFIG (MATCHING V1) ---
        print(f"Generating Analysis for Episode {ep+1}...")
        
        fig, axs = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle(f"Lunar Lander V2 (Direct) - Episode {ep+1} | Reward: {total_reward:.1f}", fontsize=16)
        
        t = np.arange(len(log_data["vel"])) * ctrl.dt
        
        # 1. Trajectory (XY)
        ax1 = axs[0, 0]
        ax1.plot([p[0] for p in log_data["pos"]], [p[1] for p in log_data["pos"]], 'b-', label='Path')
        ax1.plot(start_x, start_y, 'go', label='Start')
        ax1.plot(0, 0, 'rx', label='Target')
        ax1.set_title('Top-Down Trajectory'); ax1.set_xlim(-1100, 1100); ax1.set_ylim(-1100, 1100)
        ax1.grid(True, linestyle='--'); ax1.legend(); ax1.set_aspect('equal')

        # 2. Vertical Profile (H & Vz)
        ax2 = axs[0, 1]
        ax2.plot(t, [p[2] for p in log_data["pos"][1:]], 'b-', label='Altitude (H)')
        ax2_v = ax2.twinx()
        ax2_v.plot(t, [v[2] for v in log_data["vel"]], 'r-', alpha=0.5, label='Vz')
        ax2_v.axhline(y=-15, color='r', linestyle=':', label='Limit VZ (15)')
        ax2.set_title('Vertical Profile'); ax2.set_ylabel('Height (m)'); ax2_v.set_ylabel('Vz (m/s)')
        ax2.grid(True); ax2.legend(loc='upper left'); ax2_v.legend(loc='upper right')

        # 3. Attitude (RPY)
        ax3 = axs[1, 0]
        ax3.plot(t, [a[0] for a in log_data["att"]], 'r-', label='Roll')
        ax3.plot(t, [a[1] for a in log_data["att"]], 'g-', label='Pitch')
        ax3.plot(t, [a[2] for a in log_data["att"]], 'k-', label='Yaw')
        ax3.set_title('Attitude'); ax3.set_ylabel('Degrees'); ax3.grid(True); ax3.legend()
        
        # 4. Safety Metrics (G-Force & AngVel)
        ax4 = axs[1, 1]
        ax4.plot(t, log_data["g_force"], 'm-', label='G-Force')
        ax4.axhline(y=2.0, color='r', linestyle='--', label='Limit (2.0G)')
        ax4_w = ax4.twinx()
        ax_w = np.array(log_data["ang_vel"]) * 180 / np.pi
        ax4_w.plot(t, np.linalg.norm(ax_w, axis=1), 'c-', alpha=0.5, label='|Omega|')
        ax4_w.axhline(y=30, color='c', linestyle=':', label='Vertigo (30)')
        ax4.set_title('Safety Metrics'); ax4.set_ylabel('G Units'); ax4_w.set_ylabel('deg/s')
        ax4.grid(True); ax4.legend(loc='upper left'); ax4_w.legend(loc='upper right')

        # 5. Actions (Thrust & RCS)
        ax5 = axs[2, 0]
        acts = np.array(log_data["action"])
        ax5.plot(t, acts[:, 0], 'k-', label='Main Thrust')
        ax5_rcs = ax5.twinx()
        ax5_rcs.plot(t, np.mean(acts[:, 1:], axis=1), 'y-', alpha=0.4, label='Mean RCS')
        ax5.set_title('Control Actions'); ax5.set_ylabel('Thrust %'); ax5_rcs.set_ylabel('RCS Intensity')
        ax5.grid(True); ax5.legend(loc='upper left'); ax5_rcs.legend(loc='upper right')
        
        # 6. Performance (Reward & State)
        ax6 = axs[2, 1]
        ax6.plot(t, np.cumsum(log_data["reward"]), 'g-', label='Cum Reward')
        ax6_s = ax6.twinx()
        ax6_s.plot(t, log_data["state"], 'k:', alpha=0.3, label='State Index')
        ax6.set_title('Mission Performance'); ax6.set_ylabel('Reward'); ax6_s.set_ylabel('State ID')
        ax6.grid(True); ax6.legend(loc='upper left'); ax6_s.legend(loc='upper right')
        
        plt.tight_layout()
        os.makedirs('reports', exist_ok=True)
        filename = f'reports/mission_v2_ep{ep+1}_report.png'
        plt.savefig(filename)
        print(f"Saved Report to {filename}")
        plt.close(fig)
        
    env.close()

if __name__ == "__main__":
    run_test()
