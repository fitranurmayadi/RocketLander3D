"""Mission V3 Trajectory – LunarLander3D-aligned runner.

CLI mirrors reference:
  --episodes N   --fixed   --spawn X Y Z   --orient R P Y
  --no-render    --seed S  --max-steps N    --difficulty easy|medium|hard

Generates self-contained 3×2 mission report PNG per episode,
with reference trajectory overlay on the XY and vertical plots.
"""

import argparse
import math
import os
import time

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pybullet as pb

import rocket_lander  # noqa: F401

from experiment.mission_v3_trajectory_rocket_lander_3d_controller import (
    MissionV3TrajectoryRocketLander3DController,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def quat_to_euler(q):
    r, p, y = pb.getEulerFromQuaternion(q)
    return float(r), float(p), float(y)


# ---------------------------------------------------------------------------
# Report (3×2 subplot, reference style + trajectory overlay)
# ---------------------------------------------------------------------------

def save_report(log, ep, total_reward, start_x, start_y, out_dir, prefix, dt,
                ref_positions=None):
    os.makedirs(out_dir, exist_ok=True)
    t = np.arange(len(log["vel"])) * dt

    fig, axs = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle(
        f"Rocket Lander V3 (Trajectory) – Episode {ep} | Reward: {total_reward:.1f}",
        fontsize=16,
    )

    # 1. Top-Down Trajectory (with reference overlay)
    ax1 = axs[0, 0]
    ax1.plot([p[0] for p in log["pos"]], [p[1] for p in log["pos"]],
             "b-", label="Actual Path")
    if ref_positions and len(ref_positions) > 1:
        ax1.plot([p[0] for p in ref_positions], [p[1] for p in ref_positions],
                 "r--", alpha=0.5, label="Reference Traj")
    ax1.plot(start_x, start_y, "go", markersize=8, label="Start")
    ax1.plot(0, 0, "rx", markersize=10, label="Target")
    ax1.set_title("Top-Down Trajectory")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend()
    ax1.set_aspect("equal")

    # 2. Vertical Profile (with reference overlay)
    ax2 = axs[0, 1]
    ax2.plot(t, [p[2] for p in log["pos"][1:]], "b-", label="Altitude (m)")
    if ref_positions and len(ref_positions) > 1:
        t_ref = np.arange(len(ref_positions)) * dt
        ax2.plot(t_ref, [p[2] for p in ref_positions], "r--", alpha=0.5,
                 label="Ref Altitude")
    ax2_v = ax2.twinx()
    ax2_v.plot(t, [v[2] for v in log["vel"]], "r-", alpha=0.3, label="Vz (m/s)")
    ax2.set_title("Vertical Profile")
    ax2.set_ylabel("Height (m)")
    ax2_v.set_ylabel("Vz (m/s)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left")
    ax2_v.legend(loc="upper right")

    # 3. Attitude
    ax3 = axs[1, 0]
    ax3.plot(t, [a[0] for a in log["att"]], "r-", label="Roll")
    ax3.plot(t, [a[1] for a in log["att"]], "g-", label="Pitch")
    ax3.plot(t, [a[2] for a in log["att"]], "k-", label="Yaw")
    ax3.set_title("Attitude")
    ax3.set_ylabel("Degrees")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # 4. Safety Metrics
    ax4 = axs[1, 1]
    ax4.plot(t, log["g_force"], "m-", label="G-Force")
    ax4.axhline(y=2.0, color="r", linestyle="--", label="Limit (2.0G)")
    ax4_w = ax4.twinx()
    ang_w = np.array(log["ang_vel"]) * 180.0 / np.pi
    ax4_w.plot(t, np.linalg.norm(ang_w, axis=1), "c-", alpha=0.5, label="|Omega|")
    ax4_w.axhline(y=30, color="c", linestyle=":", label="Vertigo (30)")
    ax4.set_title("Safety Metrics")
    ax4.set_ylabel("G Units")
    ax4_w.set_ylabel("deg/s")
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc="upper left")
    ax4_w.legend(loc="upper right")

    # 5. Control Actions
    ax5 = axs[2, 0]
    acts = np.array(log["action"])
    ax5.plot(t, acts[:, 0], "k-", label="Main Thrust")
    ax5_rcs = ax5.twinx()
    ax5_rcs.plot(t, np.mean(np.abs(acts[:, 3:15]), axis=1), "y-", alpha=0.4,
                 label="Mean RCS")
    ax5.set_title("Control Actions")
    ax5.set_ylabel("Throttle")
    ax5_rcs.set_ylabel("RCS Intensity")
    ax5.grid(True, alpha=0.3)
    ax5.legend(loc="upper left")
    ax5_rcs.legend(loc="upper right")

    # 6. Tracking Error (position error from reference)
    ax6 = axs[2, 1]
    if "ref_x" in log and len(log["ref_x"]) > 0:
        pos_arr = np.array(log["pos"][1:])
        ref_arr = np.column_stack([log["ref_x"], log["ref_y"], log["ref_z"]])
        n = min(len(pos_arr), len(ref_arr))
        track_err = np.linalg.norm(pos_arr[:n] - ref_arr[:n], axis=1)
        ax6.plot(t[:n], track_err, "r-", label="3D Tracking Error (m)")
        ax6.set_title("Tracking Performance")
        ax6.set_ylabel("Error (m)")
        ax6.grid(True, alpha=0.3)
        ax6.legend()
    else:
        ax6.plot(t, np.cumsum(log["reward"]), "g-", label="Cum Reward")
        ax6.set_title("Mission Performance")
        ax6.set_ylabel("Reward")
        ax6.grid(True, alpha=0.3)
        ax6.legend()

    plt.tight_layout()
    filename = os.path.join(out_dir, f"{prefix}_report.png")
    plt.savefig(filename, dpi=150)
    print(f"Saved Report: {filename}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Rocket Lander 3D – Mission V3 (Trajectory)")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--fixed", action="store_true",
                        help="Compass Test: 4 quadrants at 200m")
    parser.add_argument("--spawn", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--orient", type=float, nargs=3, metavar=("R", "P", "Y"))
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=15000)
    parser.add_argument("--difficulty", type=str, default="easy",
                        choices=["easy", "medium", "hard"])
    args = parser.parse_args()

    render_mode = None if args.no_render else "human"
    env = gym.make("RocketLander-v0", render_mode=render_mode,
                   normalize_obs=True, randomize_spawn=False)

    if args.difficulty == "easy":
        spawn_radius = 40.0
    elif args.difficulty == "medium":
        spawn_radius = 120.0
    else:
        spawn_radius = 250.0

    dt = (1.0 / 240.0) * 4

    TOTAL_EPISODES = 4 if args.fixed else args.episodes
    base_seed = args.seed if args.seed else int(time.time())

    fixed_points = [
        [-200, -200, 55], [-200, 200, 55],
        [200, -200, 55], [200, 200, 55],
    ]
    fixed_labels = ["SW", "NW", "SE", "NE"]

    for ep in range(TOTAL_EPISODES):
        label = fixed_labels[ep % 4] if args.fixed else "RANDOM"
        print(f"\n=== EPISODE {ep + 1} / {TOTAL_EPISODES} ({label}) ===")

        if args.spawn:
            start_x, start_y, start_z = args.spawn
            if args.orient:
                r_rad = math.radians(args.orient[0])
                p_rad = math.radians(args.orient[1])
                y_rad = math.radians(args.orient[2])
            else:
                r_rad, p_rad, y_rad = 0.0, 0.0, 0.0
            print(f"Spawn (CUSTOM): X={start_x:.1f}, Y={start_y:.1f}, Z={start_z:.1f}")
        elif args.fixed:
            start_x, start_y, start_z = fixed_points[ep % 4]
            r_rad = math.radians(15)
            p_rad = math.radians(15)
            y_rad = math.radians(45)
            print(f"Spawn (FIXED-{label}): X={start_x:.1f}, Y={start_y:.1f}, Z={start_z:.1f}")
            print("Attitude (FIXED): R=15.0, P=15.0, Y=45.0")
        else:
            rng = np.random.default_rng(base_seed + ep)
            start_x = float(rng.uniform(-spawn_radius, spawn_radius))
            start_y = float(rng.uniform(-spawn_radius, spawn_radius))
            start_z = 55.0
            r_rad = float(rng.uniform(-0.15, 0.15))
            p_rad = float(rng.uniform(-0.15, 0.15))
            y_rad = float(rng.uniform(-math.pi, math.pi))
            print(f"Spawn: X={start_x:.1f}, Y={start_y:.1f}, H={start_z:.1f}")
            print(f"Attitude: R={math.degrees(r_rad):.1f}, P={math.degrees(p_rad):.1f}, "
                  f"Y={math.degrees(y_rad):.1f}")

        tilt_angle = math.sqrt(r_rad**2 + p_rad**2)
        if tilt_angle > 1e-6:
            tilt_axis = np.array([p_rad, r_rad, 0.0]) / tilt_angle
            orn = pb.getQuaternionFromAxisAngle(tilt_axis.tolist(), tilt_angle)
        else:
            orn = [0.0, 0.0, 0.0, 1.0]
        yaw_q = pb.getQuaternionFromAxisAngle([0, 0, 1], y_rad)
        orn = pb.multiplyTransforms([0, 0, 0], yaw_q, [0, 0, 0], orn)[1]

        obs, _ = env.reset(options={
            "initial_pos": [start_x, start_y, start_z],
            "initial_orn": list(orn),
        })

        ctrl = MissionV3TrajectoryRocketLander3DController(dt=dt)
        ctrl.reset()

        log = {
            "pos": [], "vel": [], "att": [], "ang_vel": [],
            "action": [], "fuel": [], "reward": [],
            "dist_h": [], "g_force": [], "state": [],
            "ref_x": [], "ref_y": [], "ref_z": [],
        }
        log["pos"].append(np.array([start_x, start_y, start_z]))

        total_reward = 0.0

        for step in range(args.max_steps):
            action, debug = ctrl.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)

            pos_real = obs[0:3] * 500.0
            vel_real = obs[7:10] * 50.0
            r, p_ang, yaw = quat_to_euler(obs[3:7])
            att_deg = np.array([math.degrees(r), math.degrees(p_ang), math.degrees(yaw)])
            ang_vel = obs[10:13] * 10.0

            log["pos"].append(pos_real.copy())
            log["vel"].append(vel_real.copy())
            log["att"].append(att_deg)
            log["ang_vel"].append(ang_vel.copy())
            log["action"].append(action.copy())
            log["fuel"].append(float(obs[18]))
            log["reward"].append(float(reward))
            log["dist_h"].append(debug["dist_h"])
            log["g_force"].append(debug["g_force"])
            log["state"].append(debug["state"])
            log["ref_x"].append(debug["ref_x"])
            log["ref_y"].append(debug["ref_y"])
            log["ref_z"].append(debug["ref_z"])

            if render_mode == "human":
                time.sleep(dt / 10.0)
                try:
                    cam_target = (float(pos_real[0]), float(pos_real[1]),
                                  float(max(1.0, pos_real[2] + 2.0)))
                    dist_cam = float(30.0 + 0.25 * max(0.0, cam_target[2]))
                    pb.resetDebugVisualizerCamera(
                        cameraDistance=dist_cam, cameraYaw=45, cameraPitch=-30,
                        cameraTargetPosition=cam_target,
                    )
                except Exception:
                    pass

            if terminated or truncated:
                print(f"Episode {ep + 1} Finished. Total Reward: {total_reward:.1f}")
                time.sleep(0.5)
                break

        out_dir = os.path.join(os.path.dirname(__file__), "outputs")
        prefix = f"v3_trajectory_ep{ep + 1}_seed{base_seed + ep}"
        save_report(log, ep + 1, total_reward, start_x, start_y, out_dir, prefix, dt,
                    ref_positions=ctrl.ref_positions)
        print(f"Mission V3 Trajectory finished. total_reward={total_reward:.1f}")

    env.close()


if __name__ == "__main__":
    main()
