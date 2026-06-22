import gymnasium as gym
import sys
import os
import numpy as np
import time
import pybullet as p
import math
from enum import Enum

# Path fix for package import
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import rocket_lander 

class ControlState(Enum):
    RECOVERY = 1
    DESCENT  = 2
    LANDING  = 3

class PID:
    def __init__(self, Kp, Ki, Kd, dt, output_limits=(None, None)):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.output_limits = output_limits
        self.prev_error = 0.0
        self.integral = 0.0
        self.setpoint = 0.0
        
    def compute(self, measurement, setpoint):
        self.setpoint = setpoint
        error = setpoint - measurement
        self.integral += error * self.dt
        self.integral = np.clip(self.integral, -1.0, 1.0)
        
        derivative = (error - self.prev_error) / self.dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        
        low, high = self.output_limits
        if low is not None: output = max(low, output)
        if high is not None: output = min(high, output)
        
        self.prev_error = error
        return output

    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0

class StagedController:
    def __init__(self, dt):
        self.dt = dt
        # 1. Height (Z) -> Throttle. Using 3500N thrust, 180kg mass.
        # Reduced gains to prevent saturation-induced oscillation
        # 1. Height (Z) -> Throttle. Action 0 -> 0.5 throttle.
        self.gravity_ff = 0.0 
        self.pid_vz = PID(0.1, 0.02, 0.1, dt, output_limits=(-1.0, 1.0))
        
        # 2. Position (XY -> Target Tilt)
        self.pid_pos_x = PID(0.1, 0.0, 0.5, dt, output_limits=(-0.1, 0.1))
        self.pid_pos_y = PID(0.1, 0.0, 0.5, dt, output_limits=(-0.1, 0.1))
        
        # 3. Attitude - CASCADED PID (Outer Angle Loop -> Inner Rate Loop)
        # Human Safety Limits: Max Rate = 1.5 rad/s
        self.rate_limit = 1.5
        
        # Outer Loops (Angle -> Target Rate) - Proactive Damping
        self.pid_pitch_outer = PID(3.0, 0.0, 0.0, dt, output_limits=(-self.rate_limit, self.rate_limit))
        self.pid_roll_outer  = PID(3.0, 0.0, 0.0, dt, output_limits=(-self.rate_limit, self.rate_limit))
        self.pid_yaw_outer   = PID(3.0, 0.0, 0.0, dt, output_limits=(-self.rate_limit, self.rate_limit))
        
        # Inner Loops (Rate Error -> RCS Torque) - Balanced for "Smooth & Pakem"
        self.pid_pitch_inner = PID(2.0, 0.0, 1.0, dt, output_limits=(-1.0, 1.0))
        self.pid_roll_inner  = PID(2.0, 0.0, 1.0, dt, output_limits=(-1.0, 1.0))
        self.pid_yaw_inner = PID(0.5, 0.0, 0.1, dt, output_limits=(-1.0, 1.0))
        
        self.state = ControlState.RECOVERY
        
    def reset(self, start_state=ControlState.RECOVERY):
        self.state = start_state
        controllers = [
            self.pid_vz, self.pid_pos_x, self.pid_pos_y, 
            self.pid_pitch_outer, self.pid_roll_outer, self.pid_yaw_outer,
            self.pid_pitch_inner, self.pid_roll_inner, self.pid_yaw_inner
        ]
        for p_ctrl in controllers:
            p_ctrl.reset()

    def get_action(self, obs):
        x, y, z = obs[0:3]
        quat = obs[3:7]
        vx, vy, vz = obs[7:10]
        wx, wy, wz = obs[10:13]
        roll, pitch, yaw = p.getEulerFromQuaternion(quat)
        dist_xy = np.sqrt(x**2 + y**2)

        # --- MISSION LOGIC ---
        target_z = 1000.0
        target_vz = 0.0
        target_yaw = 0.0
        target_pitch = 0.0
        target_roll = 0.0
        legs_cmd = -1.0 
        
        if self.state == ControlState.RECOVERY:
            # Maintain starting altitude of 1000m
            target_vz = (target_z - z) * 0.2
            target_vz = np.clip(target_vz, -5.0, 5.0)
            if abs(roll) < 0.1 and abs(pitch) < 0.1 and abs(vz) < 1.0 and abs(wx) < 0.5 and abs(wy) < 0.5:
                 print("🚀 Recovery Complete. Entering Descent/Approach.")
                 self.state = ControlState.DESCENT

        elif self.state == ControlState.DESCENT:
            # Cruise Logic: If far, stay high. If close, descend.
            if dist_xy > 200.0:
                 target_vz = 0.0 # Maintain Altitude
            else:
                 target_vz = -3.0 # Descent
                 
            target_pitch = self.pid_pos_x.compute(x, 0.0)
            target_roll = self.pid_pos_y.compute(y, 0.0)
            if z < 100:
                self.state = ControlState.LANDING
                print("🛬 Landing mode engaged.")

        elif self.state == ControlState.LANDING:
            if z > 50: target_vz = -2.5
            elif z > 20: target_vz = -1.5
            else: target_vz = -0.5
            
            target_pitch = self.pid_pos_x.compute(x, 0.0)
            target_roll = self.pid_pos_y.compute(y, 0.0)
            if z < 25: legs_cmd = 1.0 # Deploy

        # --- PID CALCULATIONS ---
        # 1. Throttle
        # compute(measurement, target) -> (target - measurement)
        throttle_out = self.pid_vz.compute(vz, target_vz)
        throttle_cmd = np.clip(self.gravity_ff + throttle_out, -1.0, 1.0)
        
        if abs(vz) > 5.0 and throttle_cmd < 0:
             # Force positive throttle if falling fast
             pass # Debugging
             
        # Extract inputs
        ang = p.getEulerFromQuaternion(obs[3:7])
        rate = obs[10:13]
        roll, pitch, yaw = ang
        target_roll, target_pitch, target_yaw = np.zeros(3) # Assuming 0 target for now in internal logic
        
        wx = rate[0]
        wy = rate[1]
        wz = rate[2] # Normal Polarity (Proven Correct via Torque Test)
             
        # 2. Attitude - Cascaded Logic
        # Outer: Angle -> Target Rate
        target_wx = self.pid_roll_outer.compute(roll, target_roll)
        target_wy = self.pid_pitch_outer.compute(pitch, target_pitch)
        target_wz = self.pid_yaw_outer.compute(yaw, target_yaw)
        
        # Inner: Rate Error -> Torque
        # Note: All axes aligned (Positive PID -> Positive Action -> Positive Torque)
        rcs_pitch_cmd = self.pid_pitch_inner.compute(wy, target_wy)
        rcs_roll_cmd  = self.pid_roll_inner.compute(wx, target_wx)
        rcs_yaw_cmd   = self.pid_yaw_inner.compute(wz, target_wz)
        

        
        # Action: [Thr, G_P, G_R, RCS_P, RCS_Y, RCS_R, Legs]
        action = np.zeros(7)
        action[0] = throttle_cmd
        action[1] = 0.0 # Locked
        action[2] = 0.0 # Locked
        action[3] = rcs_pitch_cmd # RCS Pitch
        action[4] = rcs_yaw_cmd
        action[5] = rcs_roll_cmd  # RCS Roll
        action[6] = legs_cmd
        
        # Debugging every 60 steps
        if int(time.time() * 30) % 60 == 0:
            print(f"DEBUG CTRL: Obs={obs} -> Act:{action[0:6]}")
        return action

        return action

def run_mission():
    import argparse
    parser = argparse.ArgumentParser(description="Run Rocket PID Mission")
    parser.add_argument("--render", action="store_true", help="Enable Human Rendering")
    parser.add_argument("--speed", type=float, default=1.0, help="Simulation speed multiplier (1.0 = Realtime, 0 = Fast)")
    args = parser.parse_args()

    dt = 1.0/60.0
    render_mode = "human" if args.render else None
    
    # Register environment if not already (safeguard)
    try:
        import rocket_lander
    except ImportError:
        pass
        
    env = gym.make("RocketLander-v0", render_mode=render_mode)
    
    # Configure PyBullet visualization if in human mode
    if args.render:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
    
    scenarios = [
        {"name": "1. Stabilize: Pure Pitch (45deg)", "pos": [0,0,1000], "orn": [0, 0.785, 0], "state": ControlState.RECOVERY},
        {"name": "2. Stabilize: Pure Roll (45deg)",  "pos": [0,0,1000], "orn": [0.785, 0, 0], "state": ControlState.RECOVERY},
        {"name": "3. Stabilize: Multi-Axis (45deg)", "pos": [0,0,1000], "orn": [0.5, 0.5, 0.5], "state": ControlState.RECOVERY},
        {"name": "4. Landing: High Offset (100m)",   "pos": [100.0, 100.0, 800.0], "orn": [0.0, 0.0, 0.0], "state": ControlState.DESCENT},
        {"name": "5. Challenge: Long Range V1 Classic", "pos": [-1000.0, -1000.0, 1500.0], "orn": [0.785, 0.785, 0.785], "state": ControlState.RECOVERY},
    ]

    for scen in scenarios:
        print(f"\n🌟 Starting Scenario: {scen['name']}...")
        pos_val = scen["pos"]
        orn_euler = scen["orn"]
        start_state = scen["state"]
        
        options = {
            "initial_pos": pos_val,
            "initial_orn": p.getQuaternionFromEuler(orn_euler)
        }
            
        obs, info = env.reset(options=options)
        ctrl = StagedController(dt=dt)
        ctrl.reset(start_state)
        
        loop_start = time.time()
        for step in range(2000): # Faster verification
            action = ctrl.get_action(obs)
            # Smooth Pacing
            if args.render and args.speed > 0:
                target_dt = dt / args.speed
                elapsed = time.time() - loop_start
                if elapsed < target_dt:
                    time.sleep(target_dt - elapsed)
                loop_start = time.time()
                
            obs, reward, terminated, truncated, info = env.step(action)
            
            if step % 2 == 0:
                pos = obs[0:3]
                vel = obs[7:10]
                ang = p.getEulerFromQuaternion(obs[3:7])
                rate = obs[10:13]
                dist_xy = np.sqrt(pos[0]**2 + pos[1]**2)
                state_str = f"[{ctrl.state.name}] Z:{pos[2]:5.1f} | Vz:{vel[2]:5.2f}"
                att_str = f"RPY:({ang[0]:.2f}, {ang[1]:.2f}, {ang[2]:.2f}) | W:({rate[0]:.2f}, {rate[1]:.2f}, {rate[2]:.2f})"
                # Show Targets for feedback
                tar_str = f"Target RPY:({ctrl.pid_roll_outer.setpoint:.2f}, {ctrl.pid_pitch_outer.setpoint:.2f}, {ctrl.pid_yaw_outer.setpoint:.2f})"
                print(f"{state_str} | {att_str} | {tar_str} | Thr:{action[0]:.2f} RCS:({action[3]:.1f}, {action[4]:.1f}, {action[5]:.1f})")
                
                # Deep Debug for Yaw (Index 4)
                # Yaw Error = Target - Meas
                y_err_o = ctrl.pid_yaw_outer.setpoint - ang[2]
                wz_meas = -rate[2] # Inverted perception
                y_err_i = ctrl.pid_yaw_inner.setpoint - wz_meas
                print(f"DEBUG YAW: Ang={ang[2]:.2f} Tgt={ctrl.pid_yaw_outer.setpoint:.2f} ErrO={y_err_o:.2f} -> RateTgt={ctrl.pid_yaw_inner.setpoint:.2f} RateMeas={wz_meas:.2f} ErrI={y_err_i:.2f} -> RawPID={action[4]:.2f} FinalAct={action[4]:.2f}")
                sys.stdout.flush()
                
            if terminated or truncated:
                print(f"🏁 Scenario '{scen['name']}' Complete.")
                time.sleep(2)
                break
                
    env.close()

if __name__ == "__main__":
    run_mission()
