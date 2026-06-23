"""Mission V3 (Trajectory) controller – LunarLander3D-aligned rewrite.

Two phases:
  Phase 1: Stabilize (level off, reduce angular rates)
  Phase 2: Trajectory Tracking (quintic polynomial reference)

Ported features from reference LunarLander3D mission_v3_trajectory:
  - Acceleration-based outer loop: a_ref + kp*(ref_p - p) + kd*(ref_v - v)
  - Body frame transform for forward/lateral acceleration
  - Attitude command from acceleration (target pitch/roll ∝ accel)
  - Command smoothing (alpha_cmd)
  - Heading alignment toward velocity direction
  - Deadband on attitude commands
  - Throttle ramp-up for smooth startup
"""

import math
from enum import Enum
from typing import Dict, Tuple

import numpy as np
import pybullet as pb


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def wrap_to_pi(x: float) -> float:
    if not math.isfinite(x):
        return 0.0
    return (x + math.pi) % (2 * math.pi) - math.pi


def quat_to_euler(quat_xyzw) -> Tuple[float, float, float]:
    r, pitch, yaw = pb.getEulerFromQuaternion(quat_xyzw)
    return float(r), float(pitch), float(yaw)


# ---------------------------------------------------------------------------
# Quintic Polynomial (ported from reference trajectory_planner.py)
# ---------------------------------------------------------------------------

class QuinticPolynomial:
    def __init__(self, xi, vi, ai, xf, vf, af, T):
        self.a0 = xi
        self.a1 = vi
        self.a2 = 0.5 * ai

        A = np.array([
            [T**3, T**4, T**5],
            [3 * T**2, 4 * T**3, 5 * T**4],
            [6 * T, 12 * T**2, 20 * T**3],
        ])
        b = np.array([
            xf - self.a0 - self.a1 * T - self.a2 * T**2,
            vf - self.a1 - 2 * self.a2 * T,
            af - 2 * self.a2,
        ])
        try:
            x = np.linalg.solve(A, b)
            self.a3, self.a4, self.a5 = x[0], x[1], x[2]
        except np.linalg.LinAlgError:
            self.a3 = self.a4 = self.a5 = 0.0

    def calc_point(self, t):
        xt = (self.a0 + self.a1 * t + self.a2 * t**2
              + self.a3 * t**3 + self.a4 * t**4 + self.a5 * t**5)
        vt = (self.a1 + 2 * self.a2 * t + 3 * self.a3 * t**2
              + 4 * self.a4 * t**3 + 5 * self.a5 * t**4)
        at = (2 * self.a2 + 6 * self.a3 * t
              + 12 * self.a4 * t**2 + 20 * self.a5 * t**3)
        return xt, vt, at


class Trajectory3D:
    def __init__(self, start_pos, start_vel, start_acc, end_pos, end_vel, end_acc, duration):
        self.Tx = QuinticPolynomial(start_pos[0], start_vel[0], start_acc[0],
                                    end_pos[0], end_vel[0], end_acc[0], duration)
        self.Ty = QuinticPolynomial(start_pos[1], start_vel[1], start_acc[1],
                                    end_pos[1], end_vel[1], end_acc[1], duration)
        self.Tz = QuinticPolynomial(start_pos[2], start_vel[2], start_acc[2],
                                    end_pos[2], end_vel[2], end_acc[2], duration)
        self.duration = duration

    def get_state(self, t):
        t = min(t, self.duration)
        rx, vx, ax = self.Tx.calc_point(t)
        ry, vy, ay = self.Ty.calc_point(t)
        rz, vz, az = self.Tz.calc_point(t)
        return (np.array([rx, ry, rz]),
                np.array([vx, vy, vz]),
                np.array([ax, ay, az]))


# ---------------------------------------------------------------------------
# Mission phases
# ---------------------------------------------------------------------------

class ControlState(Enum):
    STABILIZE = 1
    TRAJECTORY = 2


# ---------------------------------------------------------------------------
# PID
# ---------------------------------------------------------------------------

class PID:
    def __init__(self, kp: float, ki: float, kd: float, dt: float, limit: float = 1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.limit = limit
        self.prev_error = 0.0
        self.integral = 0.0

    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0

    def update(self, error: float, is_angular: bool = False) -> float:
        if not math.isfinite(error):
            error = 0.0
        if is_angular:
            error = wrap_to_pi(error)

        self.integral += error * self.dt
        self.integral = clamp(self.integral, -self.limit, self.limit)

        diff = error - self.prev_error
        if is_angular:
            diff = wrap_to_pi(diff)
        derivative = diff / max(1e-9, self.dt)
        self.prev_error = error

        out = self.kp * error + self.ki * self.integral + self.kd * derivative
        return clamp(out, -self.limit, self.limit)


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class MissionV3TrajectoryRocketLander3DController:
    """Trajectory-tracking controller (LunarLander3D V3 style)."""

    def __init__(self, dt: float):
        self.dt = float(dt)
        self.gravity_ff = 0.30

        # Attitude PIDs
        self.roll_pid = PID(kp=2.0, ki=0.0, kd=4.0, dt=dt, limit=1.0)
        self.pitch_pid = PID(kp=2.0, ki=0.0, kd=4.0, dt=dt, limit=1.0)
        self.yaw_pid = PID(kp=1.0, ki=0.01, kd=1.5, dt=dt, limit=1.0)

        # Trajectory tracking gains (outer loop: acceleration-based)
        self.kp_traj = 1.0  # position error → acceleration
        self.kd_traj = 3.0  # velocity error → acceleration damping

        # State
        self.state = ControlState.STABILIZE
        self.step_count = 0
        self.trajectory: Trajectory3D | None = None
        self.traj_start_step = 0

        # Smoothing
        self.target_pitch_smooth = 0.0
        self.target_roll_smooth = 0.0
        self.target_yaw_smooth = 0.0
        self.prev_vel = np.zeros(3)
        self.g_force = 1.0

        # Reference storage (for report overlay)
        self.ref_positions: list = []

        self.reset()

    def reset(self):
        self.state = ControlState.STABILIZE
        self.step_count = 0
        self.trajectory = None
        self.traj_start_step = 0
        self.target_pitch_smooth = 0.0
        self.target_roll_smooth = 0.0
        self.target_yaw_smooth = 0.0
        self.prev_vel = np.zeros(3)
        self.g_force = 1.0
        self.ref_positions = []
        for pid in [self.roll_pid, self.pitch_pid, self.yaw_pid]:
            pid.reset()

    def _generate_trajectory(self, pos, vel):
        """Generate quintic trajectory from current state to landing pad."""
        dist_h = float(np.linalg.norm(pos[0:2]))
        h = float(pos[2])
        speed = float(np.linalg.norm(vel))

        # Estimate duration based on distance and altitude
        duration = max(5.0, float(np.sqrt(dist_h**2 + h**2) / max(5.0, speed * 0.5 + 1.0)))
        duration = min(duration, 40.0)  # cap at 40s

        start_pos = pos.copy()
        start_vel = vel.copy()
        start_acc = np.array([0.0, 0.0, 0.0])

        # Target: origin at altitude = 2.0m (just above pad)
        end_pos = np.array([0.0, 0.0, 2.0])
        end_vel = np.array([0.0, 0.0, -0.3])
        end_acc = np.array([0.0, 0.0, 0.0])

        self.trajectory = Trajectory3D(
            start_pos, start_vel, start_acc,
            end_pos, end_vel, end_acc, duration
        )
        self.traj_start_step = self.step_count

        print(f"[TRAJ] Generated trajectory: Duration={duration:.1f}s | "
              f"Start={start_pos} | End={end_pos}")

    def act(self, obs: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        self.step_count += 1

        # Parse obs
        pos = obs[0:3] * 500.0
        quat = obs[3:7]
        vel = obs[7:10] * 50.0
        ang_vel = obs[10:13] * 10.0
        contacts = obs[13:17]
        h = float(obs[17] * 500.0)
        fuel = float(obs[18])

        r, pitch, yaw = quat_to_euler(quat)
        dist_h = float(np.linalg.norm(pos[0:2]))
        vz = float(vel[2])

        # G-force estimation
        accel = (vel - self.prev_vel) / max(1e-9, self.dt)
        gravity_vec = np.array([0, 0, -9.81])
        self.g_force = float(np.linalg.norm(accel - gravity_vec) / 9.81)
        self.prev_vel = vel.copy()

        # ---- State machine ----
        if self.state == ControlState.STABILIZE:
            is_stable = (abs(math.degrees(r)) < 10.0 and
                         abs(math.degrees(pitch)) < 10.0 and
                         float(np.linalg.norm(ang_vel)) < math.radians(15))
            if is_stable or self.step_count > 150:
                print("[SM] >> STABILIZED. Generating trajectory.")
                self._generate_trajectory(pos, vel)
                self.state = ControlState.TRAJECTORY

        # ---- Control ----
        target_pitch_rad = 0.0
        target_roll_rad = 0.0
        target_yaw_rad = 0.0
        throttle_target = self.gravity_ff
        ref_pos = pos.copy()

        if self.state == ControlState.STABILIZE:
            # Phase 1: Just level off and hover
            target_pitch_rad = 0.0
            target_roll_rad = 0.0
            target_yaw_rad = yaw  # hold heading
            vz_err = 0.0 - vz  # try to hover
            throttle_target = self.gravity_ff + clamp(vz_err * 0.5, -0.3, 0.3)

        elif self.state == ControlState.TRAJECTORY and self.trajectory is not None:
            # Phase 2: Trajectory tracking
            traj_t = float(self.step_count - self.traj_start_step) * self.dt

            # Throttle ramp-up for smooth startup (reference style)
            ramp = min(1.0, traj_t / 2.0)

            ref_pos, ref_vel, ref_acc = self.trajectory.get_state(traj_t)
            self.ref_positions.append(ref_pos.copy())

            # Acceleration-based outer loop (reference V3 style)
            # a_cmd = a_ref + kp*(ref_p - p) + kd*(ref_v - v)
            a_cmd = (ref_acc
                     + self.kp_traj * (ref_pos - pos)
                     + self.kd_traj * (ref_vel - vel))

            # Body frame transform (reference style)
            ax, ay, az = a_cmd[0], a_cmd[1], a_cmd[2]
            # Forward/lateral in body frame
            a_fwd = ax * math.cos(yaw) + ay * math.sin(yaw)
            a_lat = -ax * math.sin(yaw) + ay * math.cos(yaw)

            # Attitude command from acceleration (reference style)
            # target_pitch ≈ forward_accel / g   (for small angles)
            g = 9.81
            target_pitch_rad = clamp(a_fwd / g, -0.5, 0.5)
            target_roll_rad = clamp(-a_lat / g, -0.5, 0.5)

            # Yaw: face toward velocity direction (reference alpha_yaw smoothing)
            vh = float(np.linalg.norm(vel[0:2]))
            if vh > 2.0 and dist_h > 10.0:
                target_yaw_raw = math.atan2(vel[1], vel[0])
            elif dist_h > 5.0:
                target_yaw_raw = math.atan2(-pos[1], -pos[0])
            else:
                target_yaw_raw = 0.0  # align north near pad

            # Yaw smoothing
            alpha_yaw = 0.05
            yaw_err = wrap_to_pi(target_yaw_raw - self.target_yaw_smooth)
            self.target_yaw_smooth += alpha_yaw * yaw_err
            target_yaw_rad = self.target_yaw_smooth

            # Vertical thrust
            vz_err = ref_vel[2] - vel[2]
            az_cmd = ref_acc[2] + 1.0 * (ref_pos[2] - pos[2]) + 3.0 * vz_err
            throttle_target = self.gravity_ff + clamp(az_cmd * 0.05 * ramp, -0.5, 0.5)

        # Command smoothing (reference alpha_cmd)
        alpha_cmd = 0.15
        self.target_pitch_smooth = (alpha_cmd * target_pitch_rad
                                    + (1 - alpha_cmd) * self.target_pitch_smooth)
        self.target_roll_smooth = (alpha_cmd * target_roll_rad
                                   + (1 - alpha_cmd) * self.target_roll_smooth)

        # Deadband
        if abs(self.target_pitch_smooth) < 0.01:
            self.target_pitch_smooth = 0.0
        if abs(self.target_roll_smooth) < 0.01:
            self.target_roll_smooth = 0.0

        # Attitude PID
        p_err = wrap_to_pi(self.target_pitch_smooth - pitch)
        r_err = wrap_to_pi(self.target_roll_smooth - r)
        y_err = wrap_to_pi(target_yaw_rad - yaw)

        p_cmd = float(np.clip(self.pitch_pid.update(p_err), -1, 1))
        r_cmd = float(np.clip(self.roll_pid.update(r_err), -1, 1))
        y_cmd = float(np.clip(self.yaw_pid.update(y_err, is_angular=True), -1, 1))

        # Deadband on commands
        if abs(p_cmd) < 0.03:
            p_cmd = 0.0
        if abs(r_cmd) < 0.03:
            r_cmd = 0.0
        if abs(y_cmd) < 0.03:
            y_cmd = 0.0

        throttle = float(np.clip(throttle_target, 0.0, 1.0))

        # Hard cutoff on ground contact
        any_contact = bool(np.max(contacts) > 0.5 and h < 5.0)
        if any_contact:
            throttle = 0.0

        # ---- Legs ----
        legs_cmd = 1.0 if h < 6.0 else -1.0

        # ---- Build action ----
        action = np.zeros(16, dtype=np.float32)
        action[0] = throttle
        action[1] = 0.0
        action[2] = 0.0

        u_roll = float(np.clip(r_cmd, -1, 1))
        u_pitch = float(np.clip(p_cmd, -1, 1))
        u_yaw = float(np.clip(y_cmd, -1, 1))

        rcs = np.zeros(12, dtype=float)
        rcs[0] = clamp(max(0.0, u_roll), 0.0, 1.0)
        rcs[1] = clamp(max(0.0, -u_roll), 0.0, 1.0)
        rcs[2] = clamp(max(0.0, u_pitch), 0.0, 1.0)
        rcs[3] = clamp(max(0.0, -u_pitch), 0.0, 1.0)
        rcs[8] = clamp(max(0.0, u_yaw), 0.0, 1.0)
        rcs[9] = clamp(max(0.0, -u_yaw), 0.0, 1.0)
        action[3:15] = (rcs * 2.0 - 1.0).astype(np.float32)
        action[15] = float(legs_cmd)

        # Telemetry
        if self.step_count % 50 == 0:
            print(
                f"[SM] {self.step_count:<5} | St={self.state.name:<12} | "
                f"H={h:5.1f} | Vz={vz:5.2f} | Dist={dist_h:5.1f} | "
                f"Thr={throttle:.2f}"
            )

        debug = {
            "state": float(self.state.value),
            "dist_h": dist_h,
            "alt": h,
            "target_vz": float(vel[2]) if self.trajectory is None else 0.0,
            "vz": vz,
            "r_deg": math.degrees(r),
            "p_deg": math.degrees(pitch),
            "yaw_deg": math.degrees(yaw),
            "fuel": fuel,
            "g_force": self.g_force,
            "ref_x": float(ref_pos[0]),
            "ref_y": float(ref_pos[1]),
            "ref_z": float(ref_pos[2]),
        }
        return action, debug
