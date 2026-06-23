import gymnasium as gym
import rocket_lander
import pybullet as p
import numpy as np
import math
import argparse
import csv
import os
from datetime import datetime


def _rpy_to_quat(roll, pitch, yaw):
    return p.getQuaternionFromEuler([roll, pitch, yaw])


class PID:
    def __init__(self, kp, ki, kd, target=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target = target
        self.integral = 0.0
        self.prev_val = None
        self.prev_deriv = 0.0

    def compute(self, current_val, dt):
        if self.prev_val is None:
            self.prev_val = current_val

        error = self.target - current_val
        self.integral += error * dt
        self.integral = float(np.clip(self.integral, -1.0, 1.0))

        raw_deriv = -(current_val - self.prev_val) / dt
        self.prev_val = current_val

        alpha = 0.3
        filtered_deriv = (1 - alpha) * self.prev_deriv + alpha * raw_deriv
        self.prev_deriv = filtered_deriv

        return self.kp * error + self.ki * self.integral + self.kd * filtered_deriv


class RocketStabilizationPID:
    """Calibration-only PID controller: NO gimbal usage.

    Uses RCS thrusters (pitch/roll/yaw) to drive body rates and attitude
    to zero while maintaining altitude.

    action vector is length 16 (see env mapping):
      - a0 throttle in [-1,1]
      - a1,a2 gimbal commands (kept at 0)
      - a3..a14 12x RCS thrusters (kept for control)
      - a15 landing legs (kept at -1 for retracted)
    """

    def __init__(self, target_alt=500.0):
        self.MIN_MASS = 300.0
        self.MAX_MASS = 3000.0
        self.FUEL_MASS = self.MAX_MASS - self.MIN_MASS

        # Outer attitude -> target rates
        # rpy angles come from obs[3:7]
        # roll controller uses rpy[0]
        # pitch controller uses rpy[1]
        # yaw controller uses rpy[2]
        # Outer attitude -> desired rates (make correction more aggressive but damp near zero)
        # Outer attitude -> desired rates (more aggressive near zero, but damped)
        self.roll_pid = PID(kp=3.8, ki=0.0, kd=2.0, target=0.0)
        self.pitch_pid = PID(kp=3.8, ki=0.0, kd=2.0, target=0.0)
        self.yaw_pid = PID(kp=2.6, ki=0.0, kd=1.8, target=0.0)

        # Inner rates PID (faster tracking; slightly higher damping)
        self.wx_pid = PID(kp=30.0, ki=0.0, kd=8.0, target=0.0)  # roll rate (local wx)
        self.wy_pid = PID(kp=30.0, ki=0.0, kd=8.0, target=0.0)  # pitch rate (local wy)
        self.wz_pid = PID(kp=16.0, ki=0.0, kd=4.0, target=0.0)  # yaw rate (local wz)

        self.target_alt = target_alt
        self.last_diag = ""

    @staticmethod
    def _wrap_angle_to_pi(angle_rad: float) -> float:
        return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi

    def get_action(self, obs, dt):
        # Parse obs (env order): pos(3), quat(4), vel(3), ang_vel_local(3), foot(4), alt(1), fuel(1)
        # NOTE: this script uses calibration env with stable mass/fuel (see env)
        alt = obs[17]
        wx, wy, wz = obs[10], obs[11], obs[12]
        quat = obs[3:7]
        roll, pitch, yaw = p.getEulerFromQuaternion(quat)

        # Wrap-aware attitude error (target = 0)
        roll_e = self._wrap_angle_to_pi(roll)
        pitch_e = self._wrap_angle_to_pi(pitch)
        yaw_e = self._wrap_angle_to_pi(yaw)

        # Attitude -> desired rates
        roll_rate_ref = self.roll_pid.compute(roll_e, dt)
        pitch_rate_ref = self.pitch_pid.compute(pitch_e, dt)
        yaw_rate_ref = self.yaw_pid.compute(yaw_e, dt)

        # Inner rate controllers (rate tracking to ref)
        # We implement as: rate_cmd = PID(rate, target=ref)
        # For simplicity, adjust PID targets temporarily.
        self.wx_pid.target = roll_rate_ref
        self.wy_pid.target = pitch_rate_ref
        self.wz_pid.target = yaw_rate_ref

        u_roll = self.wx_pid.compute(wx, dt)
        u_pitch = self.wy_pid.compute(wy, dt)
        u_yaw = self.wz_pid.compute(wz, dt)

        # Convert continuous control effort into RCS thruster firing
        # action[3:15] are 12 thrusters defined by env.rcs_config order.
        # From env.rcs_config:
        # 0..7 are roll/pitch style (type "pr") with dir along +/-X or +/-Y
        # 8..11 are yaw style with dir along +/-X.
        # We'll follow the signs by using verify scripts instead of re-deriving.
        # Here we use a generic approach: if effort positive -> fire a matching pair;
        # thresholds avoid chattering.
        action = np.zeros(16, dtype=np.float32)

        # Main engine OFF during calibration
        action[0] = -1.0

        action[1] = 0.0  # gimbal off
        action[2] = 0.0  # gimbal off
        # --- Landing legs logic ---
        # Deploy legs (drive action[15] from retracted=-1.0 to deployed=+1.0) when near ground.
        # This ensures legs point downward when close to the terrain.
        # Tune threshold as needed.
        LEG_DEPLOY_ALT = 50.0  # meters
        action[15] = 1.0 if alt < LEG_DEPLOY_ALT else -1.0

        eps = 0.05

        # --- Roll control using pr thrusters ---
        # NOTE: Current mapping still drives roll away from zero.
        # Flip the roll channel sign (swap the thruster pair assignment).
        if u_roll > eps:
            action[3 + 3] = 1.0
            action[3 + 7] = 1.0
        elif u_roll < -eps:
            action[3 + 2] = 1.0
            action[3 + 6] = 1.0

        # --- Pitch control using pr thrusters ---
        if u_pitch > eps:
            action[3 + 0] = 1.0
            action[3 + 4] = 1.0
        elif u_pitch < -eps:
            action[3 + 1] = 1.0
            action[3 + 5] = 1.0

        # --- Yaw control using yaw thrusters ---
        # rcs_config yaw thrusters: 8..11 with type "yaw"; dirs +/-X.
        if u_yaw > eps:
            action[3 + 8] = 1.0
            action[3 + 9] = 1.0
        elif u_yaw < -eps:
            action[3 + 10] = 1.0
            action[3 + 11] = 1.0

        self.last_diag = (
            f"Alt:{alt:.1f}/{self.target_alt:.1f} "
            f"RPY(deg):({math.degrees(roll):.1f},{math.degrees(pitch):.1f},{math.degrees(yaw):.1f}) "
            f"Err(deg):({math.degrees(roll_e):.1f},{math.degrees(pitch_e):.1f},{math.degrees(yaw_e):.1f}) "
            f"Rates:[wx,wy,wz]=({wx:.3f},{wy:.3f},{wz:.3f}) "
            f"u=[roll,pitch,yaw]=({u_roll:.2f},{u_pitch:.2f},{u_yaw:.2f})"
        )
        return action


def stabilize_test(
    mode="roll",
    height=500.0,
    target_zero=True,
    no_render=False,
    out_dir="./docs/reports"
):
    print(f"\n--- PID STABILIZATION CALIBRATION: mode={mode} height={height} ---")
    os.makedirs(out_dir, exist_ok=True)

    render_mode = "human" if not no_render else None
    env = gym.make("RocketLander-v0", render_mode=render_mode, normalize_obs=False, randomize_spawn=False)

    # initial conditions: at origin horizontally, at height, with one axis rotated 90deg.
    initial_pos = [0.0, 0.0, float(height)]

    roll = pitch = yaw = 0.0
    deg45 = math.radians(45.0)
    if mode == "roll":
        roll = deg45
    elif mode == "pitch":
        pitch = deg45
    elif mode == "yaw":
        yaw = deg45
    elif mode == "full":
        roll = deg45
        pitch = deg45
        yaw = deg45
    else:
        raise ValueError("mode must be one of roll/pitch/yaw/full")

    initial_quat = _rpy_to_quat(roll, pitch, yaw)

    obs, info = env.reset(options={"initial_pos": initial_pos, "initial_orn": initial_quat})
    p.resetBaseVelocity(env.unwrapped.rocketId, [0, 0, 0], [0, 0, 0])

    controller = RocketStabilizationPID(target_alt=height)

    dt = 1.0 / 30.0
    max_steps = 2500  # ~83s at 30Hz

    # Logs for plotting
    csv_path = os.path.join(out_dir, f"pid_attitude_{mode}_h{int(height)}.csv")
    png_path = os.path.join(out_dir, f"pid_attitude_{mode}_h{int(height)}.png")

    rows = []
    t = 0.0

    try:
        for i in range(max_steps):
            action = controller.get_action(obs, dt)
            obs, reward, terminated, truncated, info = env.step(action)
            if render_mode == "human":
                env.render()

            # Collect for plotting every step (lightweight)
            quat = obs[3:7]
            r, pch, yw = p.getEulerFromQuaternion(quat)

            # Recompute wrapped errors in same way controller does
            roll_e = controller._wrap_angle_to_pi(r)
            pitch_e = controller._wrap_angle_to_pi(pch)
            yaw_e = controller._wrap_angle_to_pi(yw)

            wx, wy, wz = obs[10], obs[11], obs[12]

            # Note: u_roll/u_pitch/u_yaw tidak disimpan sebagai field terpisah, jadi kita pakai proxy:
            # - rpy dan error
            # - rates wx/wy/wz
            # - action[3:15] sum sebagai ukuran “jumlah thruster command”
            rows.append([
                t,
                float(math.degrees(r)),
                float(math.degrees(pch)),
                float(math.degrees(yw)),
                float(math.degrees(roll_e)),
                float(math.degrees(pitch_e)),
                float(math.degrees(yaw_e)),
                float(wx),
                float(wy),
                float(wz),
                float(action[0]),              # throttle cmd
                float(action[3:15].sum()),   # rcs_cmd_sum proxy
                float(action[15]),            # legs cmd
            ])

            t += dt

            if i % 10 == 0:
                print(f"T={i*dt:.1f}s {controller.last_diag} terminated={terminated}")

            if terminated or truncated:
                print("Test ended early.")
                break

            if target_zero:
                quat = obs[3:7]
                r, pch, yw = p.getEulerFromQuaternion(quat)
                # angles close to 0
                if abs(r) < math.radians(1.0) and abs(pch) < math.radians(1.0) and abs(yw) < math.radians(1.0):
                    print("SUCCESS: RPY stabilized near zero.")
                    #break

    finally:
        # Persist CSV and plot if we have data
        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "t_s",
                    "roll_deg", "pitch_deg", "yaw_deg",
                    "roll_err_deg", "pitch_err_deg", "yaw_err_deg",
                    "wx_rad_s", "wy_rad_s", "wz_rad_s",
                    "throttle_cmd", "rcs_cmd_sum", "legs_cmd"
                ])
                writer.writerows(rows)

            # Plot using matplotlib
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            if len(rows) > 0:
                data = np.array(rows, dtype=float)
                tt = data[:, 0]

                fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
                axes[0].plot(tt, data[:, 1], label="roll(deg)")
                axes[0].plot(tt, data[:, 2], label="pitch(deg)")
                axes[0].plot(tt, data[:, 3], label="yaw(deg)")
                axes[0].axhline(0.0, color="k", linewidth=0.5)
                axes[0].set_ylabel("RPY (deg)")
                axes[0].grid(True)
                axes[0].legend()

                axes[1].plot(tt, data[:, 7], label="wx (rad/s)")
                axes[1].plot(tt, data[:, 8], label="wy (rad/s)")
                axes[1].plot(tt, data[:, 9], label="wz (rad/s)")
                axes[1].axhline(0.0, color="k", linewidth=0.5)
                axes[1].set_ylabel("Rates (rad/s)")
                axes[1].grid(True)
                axes[1].legend()

                axes[2].plot(tt, data[:, 12], label="legs_cmd (a15)")
                axes[2].axhline(0.0, color="k", linewidth=0.5)
                axes[2].set_ylabel("legs_cmd")
                axes[2].set_xlabel("Time (s)")
                axes[2].grid(True)
                axes[2].legend()

                fig.suptitle(f"PID Attitude Calibration: mode={mode}, height={height}")
                fig.tight_layout()
                fig.savefig(png_path, dpi=150)
                plt.close(fig)

                print(f"Saved CSV: {csv_path}")
                print(f"Saved PNG: {png_path}")
        except Exception as e:
            print(f"[WARN] Failed to save plot/logs: {e}")

        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["roll", "pitch", "yaw", "full"], default="full")
    parser.add_argument("--height", type=float, default=500.0)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--out-dir", type=str, default="./docs/reports")
    args = parser.parse_args()

    stabilize_test(
        mode=args.mode,
        height=args.height,
        no_render=args.no_render,
        out_dir=args.out_dir
    )

