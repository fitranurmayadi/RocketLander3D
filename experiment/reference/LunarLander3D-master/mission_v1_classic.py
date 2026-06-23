# mission_v1_classic.py copied for GitHub release
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
    ORIENT             = 2 # Target 2: Point toward target before moving
    APPROACH           = 3 # Target 3: Fly to (0,0,200) & Hover
    ALIGN_NORTH        = 4 # Target 4: Yaw to 0.0
    LANDING            = 5 # Target 5: Land at (0,0,0)

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

class StateMachineController:
    def __init__(self, dt=0.01):
        self.dt = dt
        self.gravity_ff = 0.368
        # Tuning
        self.roll_pid  = PID(kp=2.0, ki=0.0, kd=10.0, dt=dt, limit=1.0)
        self.pitch_pid = PID(kp=2.0, ki=0.0, kd=10.0, dt=dt, limit=1.0)
        self.yaw_pid   = PID(kp=0.5, ki=0.01, kd=1.0, dt=dt, limit=1.0)
        self.vz_pid    = PID(kp=10.0, ki=0.1, kd=1.0, dt=dt, limit=1.0)
        self.vx_pid   = PID(kp=1.0, ki=0.01, kd=0.5, dt=dt, limit=1.0)
        self.vy_pid   = PID(kp=1.0, ki=0.01, kd=0.5, dt=dt, limit=1.0)
        self.WAYPOINT_ALT  = 200.0
        self.WAYPOINT_DIST = 10.0
        self.ATTITUDE_SAFE = 10.0
        self.start_dist_h = None
        self.move_x_smooth = 0.0
        self.move_y_smooth = 0.0
        self.last_valid_target_yaw = 0.0
        self.attitude_good_count = 0
        self.reset()
    def reset(self):
        for p in [self.roll_pid, self.pitch_pid, self.yaw_pid, self.vz_pid, self.vx_pid, self.vy_pid]:
            p.reset()
        self.state = ControlState.RECOVERY
        self.step_count = 0
        self.prev_vel = np.zeros(3)
        self.g_force = 1.0
        self.move_x_smooth = 0.0
        self.move_y_smooth = 0.0
        self.start_dist_h = None
    def reset_horizontal_pids(self):
        self.vx_pid.reset()
        self.vy_pid.reset()
        self.move_x_smooth = 0.0
        self.move_y_smooth = 0.0
    def normalize_angle(self, angle):
        while angle > math.pi: angle -= 2 * math.pi
        while angle < -math.pi: angle += 2 * math.pi
        return angle
    def update_state_machine(self, roll_deg, pitch_deg, vz, h, yaw_deg, dist_h, target_yaw_deg):
        is_stable_now = abs(roll_deg) < 10.0 and abs(pitch_deg) < 10.0
        if is_stable_now:
            self.attitude_good_count += 1
        else:
            self.attitude_good_count = 0
        is_failing = abs(roll_deg) > 45.0 or abs(pitch_deg) > 45.0
        attitude_ok = self.attitude_good_count > 20
        waypoint_reached = dist_h < self.WAYPOINT_DIST and abs(h - self.WAYPOINT_ALT) < 15.0
        yaw_err = self.normalize_angle(math.radians(target_yaw_deg - yaw_deg))
        yaw_aligned = abs(math.degrees(yaw_err)) < 5.0
        north_aligned = abs(self.normalize_angle(math.radians(0.0 - yaw_deg))) < math.radians(10.0)
        if self.state == ControlState.RECOVERY:
            if attitude_ok:
                print("[SM] >> RECOVERY COMPLETE. Orienting toward target.")
                self.state = ControlState.ORIENT
                self.reset_horizontal_pids()
        elif self.state == ControlState.ORIENT:
            if is_failing:
                self.state = ControlState.RECOVERY
            elif yaw_aligned:
                print("[SM] >> ORIENTATION COMPLETE. Fly to Waypoint.")
                self.state = ControlState.APPROACH
                self.reset_horizontal_pids()
        elif self.state == ControlState.APPROACH:
            if is_failing:
                self.state = ControlState.RECOVERY
            elif waypoint_reached:
                print("[SM] >> WAYPOINT REACHED. Aligning North.")
                self.state = ControlState.ALIGN_NORTH
        elif self.state == ControlState.ALIGN_NORTH:
            if is_failing:
                self.state = ControlState.RECOVERY
            elif north_aligned:
                print("[SM] >> ALIGNED NORTH. Starting Landing.")
                self.state = ControlState.LANDING
        elif self.state == ControlState.LANDING:
            if is_failing:
                self.state = ControlState.RECOVERY
    def compute_action(self, obs):
        if obs.shape[0] == 34: obs = obs[:17]
        pos_real = np.array([denorm_pos(obs[i], i) for i in range(3)])
        vel_real = np.array([denorm_vel(obs[i+6]) for i in range(3)])
        r, p, y = obs[3]*np.pi, obs[4]*np.pi, obs[5]*np.pi
        h = pos_real[2]; vz = vel_real[2]
        dist_h = math.sqrt(pos_real[0]**2 + pos_real[1]**2)
        if self.state in [ControlState.ORIENT, ControlState.APPROACH]:
            if dist_h > 10.0:
                self.last_valid_target_yaw = math.atan2(-pos_real[1], -pos_real[0])
            target_yaw_rad = self.last_valid_target_yaw
        elif self.state == ControlState.RECOVERY:
            target_yaw_rad = y
        else:
            target_yaw_rad = 0.0
        self.update_state_machine(math.degrees(r), math.degrees(p), vz, h, math.degrees(y), dist_h, math.degrees(target_yaw_rad))
        self.step_count += 1
        accel = (vel_real - self.prev_vel) / self.dt
        moon_g = np.array([0,0,-1.62])
        self.g_force = np.linalg.norm((accel - moon_g)) / 9.8
        self.prev_vel = vel_real.copy()
        target_vz = 0.0
        target_yaw = 0.0
        move_x, move_y = 0.0, 0.0
        if self.state == ControlState.RECOVERY:
            target_vz = 0.0
            target_yaw = y
            move_x = -vel_real[0]
            move_y = -vel_real[1]
        elif self.state == ControlState.ORIENT:
            target_vz = 0.0
            target_yaw = target_yaw_rad
            move_x = -vel_real[0]
            move_y = -vel_real[1]
        elif self.state == ControlState.APPROACH:
            if self.start_dist_h is None:
                self.start_dist_h = dist_h
            progress = np.clip((self.start_dist_h - dist_h) / max(1.0, self.start_dist_h), 0.0, 1.0)
            target_h = 1000 - (1000 - self.WAYPOINT_ALT) * progress
            alt_err = target_h - h
            target_vz = np.clip(alt_err * 0.1, -20.0, 20.0)
            target_yaw = target_yaw_rad
            norm_pos = math.sqrt(pos_real[0]**2 + pos_real[1]**2)
            if norm_pos > 1.0:
                dir_x = -pos_real[0] / norm_pos
                dir_y = -pos_real[1] / norm_pos
                speed = np.clip(norm_pos * 0.1, 2.0, 15.0)
                des_vx = dir_x * speed
                des_vy = dir_y * speed
            else:
                des_vx = des_vy = 0.0
            move_x = des_vx - vel_real[0]
            move_y = des_vy - vel_real[1]
        elif self.state == ControlState.ALIGN_NORTH:
            target_vz = np.clip((self.WAYPOINT_ALT - h)*0.1, -5, 5)
            target_yaw = 0.0
            move_x = -vel_real[0]
            move_y = -vel_real[1]
        elif self.state == ControlState.LANDING:
            target_yaw = 0.0
            des_vx = np.clip(-pos_real[0] * 0.1, -2, 2) if abs(pos_real[0]) > 0.05 else 0.0
            des_vy = np.clip(-pos_real[1] * 0.1, -2, 2) if abs(pos_real[1]) > 0.05 else 0.0
            move_x = des_vx - vel_real[0]
            move_y = des_vy - vel_real[1]
            if h > 50: target_vz = -5.0
            elif h > 10: target_vz = -2.0
            elif h > 2: target_vz = -1.0
            else: target_vz = -0.5
        a = np.zeros(21)
        r_cmd = np.clip(self.roll_pid.update(-r), -1, 1)
        p_cmd = np.clip(self.pitch_pid.update(-p), -1, 1)
        if p_cmd > 0:
            a[5]=a[9]=abs(p_cmd)
        else:
            a[4]=a[10]=abs(p_cmd)
        if r_cmd > 0:
            a[14]=a[20]=abs(r_cmd)
        else:
            a[15]=a[19]=abs(r_cmd)
        y_err = self.normalize_angle(target_yaw - y)
        y_cmd = np.clip(self.yaw_pid.update(y_err, is_angular=True), -1, 1)
        if y_cmd > 0:
            a[3]=a[8]=a[13]=a[18]=abs(y_cmd)
        else:
            a[2]=a[7]=a[12]=a[17]=abs(y_cmd)
        vz_out = self.vz_pid.update(target_vz - vz)
        a[0] = np.clip(self.gravity_ff + vz_out, 0.0, 1.0)
        alpha = 0.2
        self.move_x_smooth = alpha * move_x + (1 - alpha) * self.move_x_smooth
        self.move_y_smooth = alpha * move_y + (1 - alpha) * self.move_y_smooth
        wx = np.clip(self.vx_pid.update(self.move_x_smooth), -1, 1)
        wy = np.clip(self.vy_pid.update(self.move_y_smooth), -1, 1)
        cy = math.cos(-y); sy = math.sin(-y)
        bx = wx * cy - wy * sy
        by = wx * sy + wy * cy
        cx = np.clip(bx, -1, 1)
        cy_rcs = np.clip(by, -1, 1)
        if cx > 0: a[1]=abs(cx)
        else: a[6]=abs(cx)
        if cy_rcs > 0: a[11]=abs(cy_rcs)
        else: a[16]=abs(cy_rcs)
        if self.step_count % 50 == 0:
            print(f"[SM] St={self.state.name:<12} | H={h:6.1f} | Vz={vz:5.2f}/{target_vz:4.1f} | Y={math.degrees(y):6.1f} | Dist={dist_h:6.1f}")
        debug_info = {
            "target_vz": target_vz,
            "target_yaw": math.degrees(target_yaw),
            "target_h": self.WAYPOINT_ALT if self.state in [ControlState.APPROACH, ControlState.ALIGN_NORTH] else (0.0 if self.state == ControlState.LANDING else h),
            "dist_h": dist_h,
            "g_force": self.g_force,
            "state": self.state.value
        }
        return a, debug_info

import matplotlib.pyplot as plt
import argparse

def run_test():
    parser = argparse.ArgumentParser(description='Lunar Lander 3D Mission V1 (Classic)')
    parser.add_argument('--episodes', type=int, default=1, help='Number of episodes to run (default: 1)')
    parser.add_argument('--fixed', action='store_true', help='Compass Test: Run 4 quadrants (SW, NW, SE, NE) at 1km with 45deg tilt')
    parser.add_argument('--spawn', type=float, nargs=3, metavar=('X','Y','Z'), help='Custom spawn position')
    parser.add_argument('--orient', type=float, nargs=3, metavar=('R','P','Y'), help='Custom orientation in degrees')
    parser.add_argument('--no-render', action='store_true', help='Run without PyBullet GUI rendering')
    parser.add_argument('--no-dashboard', action='store_true', help='Disable live OSC telemetry dashboard')
    args = parser.parse_args()
    use_osc = _OSC_AVAILABLE and not args.no_dashboard
    if use_osc:
        osc_sender.init()
    render_mode = None if args.no_render else "human"
    env = gym.make('LunarLander3D-v1', render_mode=render_mode)
    TOTAL_EPISODES = 4 if args.fixed else args.episodes
    base_seed = int(time.time())
    fixed_points = [[-1000,-1000,1000],[-1000,1000,1000],[1000,-1000,1000],[1000,1000,1000]]
    fixed_labels = ["SW","NW","SE","NE"]
    for ep in range(TOTAL_EPISODES):
        label = fixed_labels[ep % 4] if args.fixed else "RANDOM"
        print(f"\n=== EPISODE {ep+1} / {TOTAL_EPISODES} ({label}) ===")
        if use_osc:
            osc_sender.send_episode(ep + 1, label)
        if args.spawn:
            start_x,start_y,start_z = args.spawn
            r,p,y = [math.radians(v) for v in args.orient] if args.orient else [0,0,0]
            print(f"Spawn (CUSTOM): X={start_x:.1f}, Y={start_y:.1f}, Z={start_z:.1f}")
        elif args.fixed:
            start_x,start_y,start_z = fixed_points[ep % 4]
            r,p,y = math.radians(45),math.radians(45),math.radians(45)
            print(f"Spawn (FIXED-{label}): X={start_x:.1f}, Y={start_y:.1f}, Z={start_z:.1f}")
            print("Attitude (FIXED): R=45.0, P=45.0, Y=45.0")
        else:
            np.random.seed(base_seed + ep)
            start_x = np.random.uniform(-900,900)
            start_y = np.random.uniform(-900,900)
            start_z = 1000.0
            r = np.random.uniform(-0.5,0.5)
            p = np.random.uniform(-0.5,0.5)
            y = np.random.uniform(-3.14,3.14)
            print(f"Spawn: X={start_x:.1f}, Y={start_y:.1f}, H={start_z:.1f}")
            print(f"Attitude: R={math.degrees(r):.1f}, P={math.degrees(p):.1f}, Y={math.degrees(y):.1f}")
        obs, _ = env.reset(options={"initial_pos":[start_x,start_y,start_z],"initial_orient":[r,p,y]})
        ctrl = StateMachineController(dt=env.unwrapped.sim_time)
        log_data = {"pos":[],"vel":[],"att":[],"ang_vel":[],"action":[],"fuel":[],"reward":[],"target_vz":[],"target_yaw":[],"dist_h":[],"g_force":[],"state":[]}
        total_reward = 0
        log_data["pos"].append(np.array([start_x,start_y,start_z]))
        for i in range(100000):
            action,debug = ctrl.compute_action(obs)
            obs,reward,done,trunc,_ = env.step(action)
            total_reward += reward
            current = obs[:17] if obs.shape[0]==34 else obs
            pos_real = np.array([denorm_pos(current[k],k) for k in range(3)])
            vel_real = np.array([denorm_vel(current[k+6]) for k in range(3)])
            att_real = np.array([current[3]*180,current[4]*180,current[5]*180])
            ang_real = np.array([current[9],current[10],current[11]])
            fuel = current[12]
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
            log_data["g_force"].append(debug["g_force"])
            log_data["state"].append(debug["state"])
            # --- OSC telemetry (every step ~10ms) ---
            if use_osc:
                cum_r = float(np.sum(log_data["reward"]))
                osc_sender.send_state(i, float(pos_real[2]), float(vel_real[2]),
                                      float(debug["dist_h"]), float(reward),
                                      int(debug["state"]), cum_r)
                osc_sender.send_attitude(float(att_real[0]), float(att_real[1]), float(att_real[2]))
                osc_sender.send_velocity(float(vel_real[0]), float(vel_real[1]), float(vel_real[2]))
                osc_sender.send_action(float(action[0]), float(np.mean(action[1:])), float(debug["g_force"]))
            if not args.no_render:
                time.sleep(ctrl.dt/10.0)
            if done or trunc:
                print(f"Episode {ep+1} Finished. Total Reward: {total_reward:.1f}")
                time.sleep(1.0)
                break
        print(f"Generating Analysis for Episode {ep+1}...")
        fig, axs = plt.subplots(3,2,figsize=(16,18))
        fig.suptitle(f"Lunar Lander V1 (Classic) - Episode {ep+1} | Reward: {total_reward:.1f}", fontsize=16)
        t = np.arange(len(log_data["vel"])) * ctrl.dt
        ax1 = axs[0,0]
        ax1.plot([p[0] for p in log_data["pos"]],[p[1] for p in log_data["pos"]],'b-', label='Path')
        ax1.plot(start_x,start_y,'go', label='Start')
        ax1.plot(0,0,'rx', label='Target')
        ax1.set_title('Top-Down Trajectory'); ax1.set_xlim(-1100,1100); ax1.set_ylim(-1100,1100)
        ax1.grid(True,linestyle='--'); ax1.legend(); ax1.set_aspect('equal')
        
        ax2 = axs[0,1]
        ax2.plot(t,[p[2] for p in log_data["pos"][1:]],'b-', label='Altitude (H)')
        ax2_v = ax2.twinx()
        ax2_v.plot(t,[v[2] for v in log_data["vel"]],'r-',alpha=0.5, label='Vz')
        ax2_v.axhline(y=-15,color='r',linestyle=':', label='Limit VZ (15)')
        ax2.set_title('Vertical Profile'); ax2.set_ylabel('Height (m)'); ax2_v.set_ylabel('Vz (m/s)')
        ax2.grid(True); ax2.legend(loc='upper left'); ax2_v.legend(loc='upper right')
        
        ax3 = axs[1,0]
        ax3.plot(t,[a[0] for a in log_data["att"]],'r-', label='Roll')
        ax3.plot(t,[a[1] for a in log_data["att"]],'g-', label='Pitch')
        ax3.plot(t,[a[2] for a in log_data["att"]],'k-', label='Yaw')
        ax3.set_title('Attitude'); ax3.set_ylabel('Degrees'); ax3.grid(True); ax3.legend()
        
        ax4 = axs[1,1]
        ax4.plot(t,log_data["g_force"],'m-', label='G-Force')
        ax4.axhline(y=2.0,color='r',linestyle='--', label='Limit (2.0G)')
        ax4_w = ax4.twinx()
        ax_w = np.array(log_data["ang_vel"]) * 180 / np.pi
        ax4_w.plot(t,np.linalg.norm(ax_w,axis=1),'c-', alpha=0.5, label='|Omega|')
        ax4_w.axhline(y=30,color='c',linestyle=':', label='Vertigo (30)')
        ax4.set_title('Safety Metrics'); ax4.set_ylabel('G Units'); ax4_w.set_ylabel('deg/s')
        ax4.grid(True); ax4.legend(loc='upper left'); ax4_w.legend(loc='upper right')
        
        ax5 = axs[2,0]
        acts = np.array(log_data["action"])
        ax5.plot(t,acts[:,0],'k-', label='Main Thrust')
        ax5_rcs = ax5.twinx()
        ax5_rcs.plot(t,np.mean(acts[:,1:],axis=1),'y-', alpha=0.4, label='Mean RCS')
        ax5.set_title('Control Actions'); ax5.set_ylabel('Thrust %'); ax5_rcs.set_ylabel('RCS Intensity')
        ax5.grid(True); ax5.legend(loc='upper left'); ax5_rcs.legend(loc='upper right')
        
        ax6 = axs[2,1]
        ax6.plot(t,np.cumsum(log_data["reward"]),'g-', label='Cum Reward')
        ax6_s = ax6.twinx()
        ax6_s.plot(t,log_data["state"],'k:', alpha=0.3, label='State Index')
        ax6.set_title('Mission Performance'); ax6.set_ylabel('Reward'); ax6_s.set_ylabel('State ID')
        ax6.grid(True); ax6.legend(loc='upper left'); ax6_s.legend(loc='upper right')
        plt.tight_layout()
        os.makedirs('reports', exist_ok=True)
        filename = f'reports/mission_v1_ep{ep+1}_report.png'
        plt.savefig(filename)
        print(f"Saved Report to {filename}")
        plt.close(fig)
    env.close()

if __name__ == "__main__":
    run_test()
