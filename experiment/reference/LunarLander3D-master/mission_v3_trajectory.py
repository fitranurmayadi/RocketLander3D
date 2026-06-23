import math
import time
import argparse
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
from trajectory_planner import Trajectory3D
import gymnasium as gym
import os, sys
# Ensure project root is in PYTHONPATH for local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    import osc_sender
    _OSC_AVAILABLE = True
except ImportError:
    _OSC_AVAILABLE = False

# Safe import of lunar_lander_3d (may be optional)
try:
    import lunar_lander_3d
except ImportError:
    lunar_lander_3d = None  # Placeholder if the module is unavailable

import numpy as np


class PID:
    def __init__(self, kp, ki, kd, limit):
        self.kp, self.ki, self.kd, self.limit = kp, ki, kd, limit
        self.pe, self.fd, self.integral = 0.0, 0.0, 0.0

    def update(self, e, edot=None, dt=0.01):
        self.integral += e * dt
        self.integral = np.clip(self.integral, -self.limit, self.limit)
        
        if edot is None:
            rd = (e - self.pe)/dt
            self.fd = 0.1*rd + 0.9*self.fd
        else:
            self.fd = edot
            
        self.pe = e
        return np.clip(self.kp*e + self.ki*self.integral + self.kd*self.fd, -self.limit, self.limit)

class MissionController:
    def __init__(self, iy):
        self.dt = 0.01
        # Trajectory Authority (High Precision)
        self.kp_pos = 1.20
        self.kd_vel = 1.80
        self.z_pid = PID(10.0, 0.1, 5.0, 1.0) # Aggressive Z for heavy lander
        
        # Inner Loop: Attitude (Stiff & Damped)
        self.r_pid = PID(8.0, 0.0, 20.0, 1.0)
        self.p_pid = PID(8.0, 0.0, 20.0, 1.0)
        self.y_pid_inner = PID(5.0, 0.0, 10.0, 1.0)
        
        self.db = 0.1 # Larger deadband for smoother control
        self.target_p, self.target_r, self.target_y = 0.0, 0.0, float(iy)
        self.alpha_cmd = 0.5 # Faster attitude command response
        self.alpha_yaw = 0.05

    def compute(self, obs, ref_p, ref_v, ref_a, t):
        if obs.shape[0] == 34: obs = obs[17:]
        cp = obs[0:3]*2000.0; cv = obs[6:9]*200.0; att = obs[3:6]*np.pi # Radians!
        
        # 1. World Frame Acceleration Commands
        ax = ref_a[0] + self.kp_pos*(ref_p[0]-cp[0]) + self.kd_vel*(ref_v[0]-cv[0])
        ay = ref_a[1] + self.kp_pos*(ref_p[1]-cp[1]) + self.kd_vel*(ref_v[1]-cv[1])
        az = ref_a[2] + self.z_pid.update(ref_p[2]-cp[2], ref_v[2]-cv[2])
        
        # 2. Body Frame Transform
        c, s = np.cos(att[2]), np.sin(att[2])
        ah =  ax*c + ay*s # Forward
        al = -ax*s + ay*c # Left
        
        # 3. Validated Mapping: +Forward Accel -> Nose UP (+Pitch)
        #                     +Left Accel -> Roll LEFT (-Roll)
        limit = 0.15 # 8.5 deg safety
        tp_raw = np.clip(ah * 0.62, -limit, limit)
        tr_raw = np.clip(-al * 0.62, -limit, limit)
        
        # Command Smoothing
        self.target_p = self.alpha_cmd*tp_raw + (1-self.alpha_cmd)*self.target_p
        self.target_r = self.alpha_cmd*tr_raw + (1-self.alpha_cmd)*self.target_r
        
        # 4. Heading Alignment
        if np.linalg.norm(ref_v[0:2]) > 0.1:
            ry = np.arctan2(ref_v[1], ref_v[0])
            self.target_y += self.alpha_yaw*((ry - self.target_y + np.pi)%(2*np.pi)-np.pi)
        
        # 5. Inner Loop (Attitude)
        pr = self.r_pid.update(self.target_r - att[0])
        qr = self.p_pid.update(self.target_p - att[1])
        yr = self.y_pid_inner.update((self.target_y - att[2] + np.pi)%(2*np.pi)-np.pi)
        
        # 6. Action Mixer
        cs = max(0.1, np.cos(att[0])*np.cos(att[1]))
        action = np.zeros(21)
        # Feedforward: Hover throttle ~0.41
        action[0] = np.clip((0.41 + az*0.25)/cs, 0.0, 1.0) * min(1.0, t)
        
        if abs(pr)<self.db: pr=0
        if abs(qr)<self.db: qr=0
        if abs(yr)<self.db: yr=0
        
        if qr>0: action[5]=action[9]=abs(qr)
        elif qr<0: action[4]=action[10]=abs(qr)
        if pr>0: action[14]=action[20]=abs(pr)
        elif pr<0: action[15]=action[19]=abs(pr)
        if yr>0: action[3]=action[8]=action[13]=action[18]=abs(yr)
        else: action[2]=action[7]=action[12]=action[17]=abs(yr)
        
        return action, cp, cv, att

def report(log, name='mission_v3_ep1_report.png'):
    print(f"Generating Performance Report: {name}")
    print("Generating Final Mission Performance Report...")
    fig, axs = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle("Lunar Lander V3 - Trajectory Performance Analysis", fontsize=16)
    
    t = np.arange(len(log["pos"])) * 0.01
    p = np.array(log["pos"])
    r = np.array(log["ref"])
    v = np.array(log["vel"])
    a = np.array(log["att"]) * 180 / np.pi
    
    # 1. Top-Down Trajectory
    ax1 = axs[0, 0]
    ax1.plot(p[:,0], p[:,1], 'b-', label='Actual Path')
    ax1.plot(r[:,0], r[:,1], 'r--', alpha=0.5, label='Reference')
    ax1.plot(p[0,0], p[0,1], 'go', label='Start')
    ax1.plot(0, 0, 'rx', label='Target')
    ax1.set_title('Top-Down Trajectory (XY)'); ax1.grid(True); ax1.legend()
    ax1.set_aspect('equal')
    ax1.set_xlim(-1000, 1000)
    ax1.set_ylim(-1000, 1000)
    
    # 2. Vertical Profile
    ax2 = axs[0, 1]
    ax2.plot(t, p[:,2], 'b-', label='Altitude')
    ax2.plot(t, r[:,2], 'r--', alpha=0.5, label='Ref Alt')
    ax2_v = ax2.twinx()
    ax2_v.plot(t, v[:,2], 'g-', alpha=0.3, label='Vz')
    ax2.set_title('Vertical Profile'); ax2.set_ylabel('Altitude (m)'); ax2_v.set_ylabel('Vz (m/s)')
    ax2.grid(True); ax2.legend(loc='upper left'); ax2_v.legend(loc='upper right')
    
    # 3. Attitude (RPY)
    ax3 = axs[1, 0]
    ax3.plot(t, a[:,0], 'r-', label='Roll')
    ax3.plot(t, a[:,1], 'g-', label='Pitch')
    ax3.plot(t, a[:,2], 'k-', label='Yaw')
    ax3.plot(t, np.array(log["ry"])*180/np.pi, 'k--', alpha=0.3, label='Target Yaw')
    ax3.set_title('Orientation (Attitude)'); ax3.set_ylabel('Degrees'); ax3.grid(True); ax3.legend()
    
    # 4. Safety Metrics (G-Force & Omega)
    ax4 = axs[1, 1]
    ax4.plot(t, log["g_force"], 'm-', label='G-Force')
    ax4.axhline(y=2.0, color='r', linestyle='--', alpha=0.5, label='2.0G Limit')
    ax4_o = ax4.twinx()
    omega = np.linalg.norm(np.array(log["ang_vel"]), axis=1) * 180 / np.pi
    ax4_o.plot(t, omega, 'c-', alpha=0.4, label='|Omega|')
    ax4_o.axhline(y=30, color='c', linestyle=':', label='30 deg/s Vertigo')
    ax4.set_title('Safety Metrics'); ax4.set_ylabel('G-Force'); ax4_o.set_ylabel('deg/s')
    ax4.grid(True); ax4.legend(loc='upper left'); ax4_o.legend(loc='upper right')
    
    # 5. Control Actions
    ax5 = axs[2, 0]
    acts = np.array(log["act"])
    ax5.plot(t, acts[:,0], 'k-', label='Main Thrust')
    ax5_rcs = ax5.twinx()
    ax5_rcs.plot(t, np.mean(acts[:,1:], axis=1), 'y-', alpha=0.5, label='Mean RCS')
    ax5.set_title('Control Actions'); ax5.set_ylabel('Thrust %'); ax5_rcs.set_ylabel('RCS Intensity')
    ax5.grid(True); ax5.legend(loc='upper left'); ax5_rcs.legend(loc='upper right')
    
    # 6. Tracking Performance
    ax6 = axs[2, 1]
    err = np.linalg.norm(p - r, axis=1)
    ax6.plot(t, err, 'b-', label='3D Tracking Error')
    ax6_r = ax6.twinx()
    ax6_r.plot(t, np.cumsum(log["reward"]), 'g-', alpha=0.3, label='Cum Reward')
    ax6.set_title('Mission Performance'); ax6.set_ylabel('Error (m)'); ax6_r.set_ylabel('Total Reward')
    ax6.grid(True); ax6.legend(loc='upper left'); ax6_r.legend(loc='upper right')
    
    plt.tight_layout()
    os.makedirs('reports', exist_ok=True)
    plt.savefig(name)
    plt.close()


def run():
    parser = argparse.ArgumentParser(description='Lunar Lander 3D Mission V3 (Trajectory Mastery)')
    parser.add_argument('--episodes', type=int, default=1, help='Number of episodes to run')
    parser.add_argument('--fixed', action='store_true', help='Compass Test: Run 4 quadrants (SW, NW, SE, NE) at 1km with 45deg tilt')
    parser.add_argument('--spawn', type=float, nargs=3, metavar=('X','Y','Z'), help='Custom spawn position')
    parser.add_argument('--orient', type=float, nargs=3, metavar=('R','P','Y'), help='Custom orientation in degrees')
    parser.add_argument('--no-render', action='store_true', help='Disable GUI rendering')
    parser.add_argument('--no-dashboard', action='store_true', help='Disable live OSC telemetry dashboard')
    args = parser.parse_args()
    use_osc = _OSC_AVAILABLE and not args.no_dashboard
    if use_osc:
        osc_sender.init()
    
    render = None if args.no_render else "human"
    env = gym.make("LunarLander3D-v1", render_mode=render)
    
    TOTAL_EPISODES = 4 if args.fixed else args.episodes
    base_seed = int(time.time())
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
        
        if args.spawn:
            sp = np.array(args.spawn)
            sa = [math.radians(v) for v in args.orient] if args.orient else [0, 0, 0]
            print(f"Spawn (CUSTOM): X={sp[0]:.1f}, Y={sp[1]:.1f}, Z={sp[2]:.1f}")
        elif args.fixed:
            sp = np.array(fixed_points[ep % 4])
            sa = [np.radians(45), np.radians(45), np.radians(45)]
            print(f"Spawn (FIXED-{label}): X={sp[0]:.1f}, Y={sp[1]:.1f}, Z={sp[2]:.1f}")
            print("Attitude (FIXED): R=45.0, P=45.0, Y=45.0")
        else:
            np.random.seed(base_seed + ep)
            sp = np.array([np.random.uniform(-900, 900), np.random.uniform(-900, 900), 1000.0])
            sa = [np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5), np.random.uniform(-3.14, 3.14)]
            print(f"Spawn: X={sp[0]:.1f}, Y={sp[1]:.1f}, H={sp[2]:.1f}")
            print(f"Attitude: R={math.degrees(sa[0]):.1f}, P={math.degrees(sa[1]):.1f}, Y={math.degrees(sa[2]):.1f}")
            
        obs, _ = env.reset(options={"spawn_pos": sp.tolist(), "spawn_att": sa})
        
        # --- INITIALIZATION ---
        iy = sa[2]
        ctrl = MissionController(iy)
        duration = 200.0 
        target_h = 9.9 
        
        log = {
            "pos":[], "ref":[], "att":[], "vel":[], "ang_vel":[], 
            "act":[], "ry":[], "reward":[], "g_force":[]
        }
        
        print(f"COMMENCING MISSION {label}: Start={sp}")
        
        # --- PHASE 1: PRE-FLIGHT STABILIZATION ---
        print("PHASE 1: ACTIVE STABILIZATION (Recovering from tumble...)")
        stabilized = False
        for i in range(1000): # Max 10s
            # Hold initial position, recover attitude
            action, cp, cv, att = ctrl.compute(obs, sp, np.zeros(3), np.zeros(3), i*0.01)
            obs, reward, done, trunc, _ = env.step(action)
            
            # Telemetry for logging and exit check
            obs_data = obs[17:] if obs.shape[0] == 34 else obs
            ang_vel = obs_data[9:12] * 10.0
            
            # Log stabilization data
            log["pos"].append(cp); log["ref"].append(sp); log["att"].append(att)
            log["vel"].append(cv); log["ang_vel"].append(ang_vel); log["act"].append(action)
            log["ry"].append(ctrl.target_y); log["reward"].append(reward); log["g_force"].append(0.0)
            
            if render == "human": time.sleep(0.002)
            
            # Exit Check: Level enough and slow enough
            att_err = np.linalg.norm(att[:2]) * 180 / np.pi
            vel_err = np.linalg.norm(cv)
            if att_err < 2.0 and vel_err < 0.5:
                print(f"STABILIZED at T={i*0.01:.2f}s: AttErr={att_err:.1f}deg, Vel={vel_err:.2f}m/s")
                stabilized = True
                break
        
        if not stabilized: print("WARNING: Proceeding to Phase 2 without full stabilization.")
        
        # --- PHASE 2: DYNAMIC TRAJECTORY GENERATION ---
        # Re-plan from ACTUAL position after stabilization
        current_p = cp.copy()
        print(f"PHASE 2: TRAJECTORY START from {current_p}")
        traj = Trajectory3D(current_p, cv, [0,0,0], [0,0,target_h+1.0], [0,0,0], [0,0,0], duration)
        
        prev_vel = cv.copy()
        landing_mode = False
        for i in range(int((duration + 30) * 100)): 
            t = i * 0.01
            if not landing_mode:
                ref_p, ref_v, ref_a = traj.get_state(t)
                if t > duration: landing_mode = True
            else:
                ref_p = np.array([0.0, 0.0, target_h + 0.1]); ref_v = np.array([0.0, 0.0, -0.5]); ref_a = np.zeros(3)
                
            action, cp, cv, att = ctrl.compute(obs, ref_p, ref_v, ref_a, t)
            obs, reward, done, trunc, _ = env.step(action)
            
            # Telemetry
            obs_data = obs[17:] if obs.shape[0] == 34 else obs
            ang_vel = obs_data[9:12] * 10.0; contacts = obs_data[13:17]
            accel = (cv - prev_vel) / 0.01
            is_contact = np.any(contacts > 0.001) or done
            g_force = np.clip(np.linalg.norm(accel - [0,0,-1.62])/9.8 if not is_contact else log["g_force"][-1], 0, 2)
            prev_vel = cv.copy()
            
            if render == "human": time.sleep(0.002) 
            
            log["pos"].append(cp); log["ref"].append(ref_p); log["att"].append(att)
            log["vel"].append(cv); log["ang_vel"].append(ang_vel); log["act"].append(action)
            log["ry"].append(ctrl.target_y); log["reward"].append(reward); log["g_force"].append(g_force)
            # --- OSC telemetry (every step ~10ms) ---
            if use_osc:
                cum_r = float(sum(log["reward"]))
                dist_h = float(np.linalg.norm(cp[:2]))
                att_deg = [float(a * 180 / np.pi) for a in att[:3]]
                osc_sender.send_state(i, float(cp[2]), float(cv[2]), dist_h,
                                      float(reward), 2, cum_r)
                osc_sender.send_attitude(att_deg[0], att_deg[1], att_deg[2])
                osc_sender.send_velocity(float(cv[0]), float(cv[1]), float(cv[2]))
                osc_sender.send_action(float(action[0]), float(np.mean(action[1:])), float(g_force))
            
            if np.any(contacts > 0.1) and cp[2] < (target_h + 1.0):
                print(f"--- SUCCESSFUL TOUCHDOWN AT T={t:.2f} ---")
                break
                
            if i % 2000 == 0:
                print(f"T={t:5.1f} | Pos={cp} | Lag={np.linalg.norm(ref_p-cp):.2f}m")
                
        report_name = f"reports/mission_v3_ep{ep+1}_report.png"
        report(log, report_name)
        print(f"EPISODE {ep+1} ANALYSIS COMPLETE: {report_name}")
        
    env.close()

if __name__ == "__main__":
    run()
