"""Mission V2 (Direct) controller – LunarLander3D-aligned rewrite.

State machine:
  RECOVERY → ORIENT → TRANSIT → LANDING

Ported features from reference LunarLander3D mission_v2_direct:
  - Authority separation: cruise tilt-only vs approach tilt+RCS
  - Target pitch/roll smoothing with rate limiting
  - Thrust slew rate limiting
  - cos-safe thrust compensation
  - Angular comfort cap
  - Command smoothing (LPF)
  - Deadband on commands and RCS
  - Hard engine cutoff on contact
  - Safety envelope (MAX_TILT, MAX_G, MAX_W, MAX_VH, MAX_VZ)
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
# Mission phases
# ---------------------------------------------------------------------------

class ControlState(Enum):
    RECOVERY = 1
    ORIENT = 2
    TRANSIT = 3
    LANDING = 4


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

class MissionV2DirectRocketLander3DController:
    """Direct 4-phase controller with authority separation (LunarLander3D V2 style)."""

    def __init__(self, dt: float):
        self.dt = float(dt)
        self.gravity_ff = 0.30

        # Attitude PIDs (anti-oscillation tuned)
        self.roll_pid = PID(kp=1.5, ki=0.0, kd=2.0, dt=dt, limit=1.0)
        self.pitch_pid = PID(kp=1.5, ki=0.0, kd=2.0, dt=dt, limit=1.0)
        self.yaw_pid = PID(kp=1.0, ki=0.01, kd=1.0, dt=dt, limit=1.0)

        # Vertical PID
        self.vz_pid = PID(kp=4.0, ki=0.1, kd=1.0, dt=dt, limit=1.0)

        # Horizontal PIDs (soft for smooth transit)
        self.vx_pid = PID(kp=0.1, ki=0.0, kd=0.2, dt=dt, limit=1.0)
        self.vy_pid = PID(kp=0.1, ki=0.0, kd=0.2, dt=dt, limit=1.0)

        # Safety envelope
        self.MAX_TILT = 0.5   # rad limit for approach
        self.MAX_G = 2.0
        self.MAX_W = math.radians(30.0)  # 30 deg/s
        self.MAX_VH = 8.0     # m/s horizontal speed cap
        self.MAX_VZ = 10.0
        self.THRUST_SLEW = 0.05  # max thrust change per step

        # State
        self.state = ControlState.RECOVERY
        self.step_count = 0
        self.last_valid_target_yaw = 0.0

        # Smoothing
        self.target_pitch_smooth = 0.0
        self.target_roll_smooth = 0.0
        self.bx_rcs_smooth = 0.0
        self.by_rcs_smooth = 0.0
        self.p_cmd_smooth = 0.0
        self.r_cmd_smooth = 0.0
        self.y_cmd_smooth = 0.0
        self.prev_thrust = 0.30
        self.prev_vel = np.zeros(3)
        self.g_force = 1.0

        self.reset()

    def reset(self):
        self.state = ControlState.RECOVERY
        self.step_count = 0
        self.last_valid_target_yaw = 0.0
        self.target_pitch_smooth = 0.0
        self.target_roll_smooth = 0.0
        self.bx_rcs_smooth = 0.0
        self.by_rcs_smooth = 0.0
        self.p_cmd_smooth = 0.0
        self.r_cmd_smooth = 0.0
        self.y_cmd_smooth = 0.0
        self.prev_thrust = 0.30
        self.prev_vel = np.zeros(3)
        self.g_force = 1.0
        for pid in [self.roll_pid, self.pitch_pid, self.yaw_pid,
                    self.vz_pid, self.vx_pid, self.vy_pid]:
            pid.reset()

    def _reset_horizontal(self):
        self.vx_pid.reset()
        self.vy_pid.reset()

    # ----- State machine -----
    def _update_state_machine(self, r: float, p_ang: float, vz: float,
                              h: float, yaw: float, dist_h: float, any_contact: bool):
        r_deg = abs(math.degrees(r))
        p_deg = abs(math.degrees(p_ang))
        is_failing = r_deg > 30.0 or p_deg > 30.0

        if self.state == ControlState.RECOVERY:
            if r_deg < 10.0 and p_deg < 10.0 and abs(vz) < 5.0:
                print("[SM] >> RECOVERY COMPLETE.")
                self.state = ControlState.ORIENT

        elif self.state == ControlState.ORIENT:
            y_err = abs(math.degrees(wrap_to_pi(self.last_valid_target_yaw - yaw)))
            if is_failing:
                self.state = ControlState.RECOVERY
            elif y_err < 10.0:
                print("[SM] >> ORIENTED. Start Transit.")
                self.state = ControlState.TRANSIT
                self._reset_horizontal()

        elif self.state == ControlState.TRANSIT:
            if is_failing:
                self.state = ControlState.RECOVERY
            elif (dist_h < 5.0 and h < 30.0) or (any_contact and h < 5.0):
                print("[SM] >> TRANSIT DONE. Landing.")
                self.state = ControlState.LANDING
                self._reset_horizontal()

        elif self.state == ControlState.LANDING:
            if is_failing:
                self.state = ControlState.RECOVERY
            elif dist_h > 25.0:
                print(f"[SM] >> LOST TARGET. Abort to TRANSIT. Dist={dist_h:.1f}")
                self.state = ControlState.TRANSIT
                self._reset_horizontal()

    # ----- Main act -----
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
        vz = float(vel[2])
        dist_h = float(np.linalg.norm(pos[0:2]))

        # Target yaw
        if self.state in [ControlState.ORIENT, ControlState.TRANSIT]:
            if dist_h > 10.0:
                self.last_valid_target_yaw = math.atan2(-pos[1], -pos[0])
            target_yaw_rad = self.last_valid_target_yaw
        elif self.state == ControlState.RECOVERY:
            target_yaw_rad = yaw
        else:
            target_yaw_rad = 0.0

        # Contact detection (with altitude sanity check)
        any_contact = bool(np.max(contacts) > 0.5 and h < 5.0)

        # State machine
        self._update_state_machine(r, pitch, vz, h, yaw, dist_h, any_contact)

        # G-force estimation
        accel = (vel - self.prev_vel) / max(1e-9, self.dt)
        gravity_vec = np.array([0, 0, -9.81])
        self.g_force = float(np.linalg.norm(accel - gravity_vec) / 9.81)
        self.prev_vel = vel.copy()

        # ---- Control targets ----
        target_vz = 0.0
        target_yaw = target_yaw_rad
        move_x, move_y = 0.0, 0.0

        if self.state == ControlState.RECOVERY:
            target_vz = 0.0
            target_yaw = yaw
            move_x, move_y = -vel[0], -vel[1]

        elif self.state == ControlState.ORIENT:
            target_vz = 0.0
            target_yaw = target_yaw_rad
            move_x = -vel[0]
            move_y = -vel[1]

        elif self.state == ControlState.TRANSIT:
            # Altitude descent
            if h > 50:
                target_vz = -5.0
            else:
                target_vz = -2.0
            target_yaw = target_yaw_rad
            des_vx = float(np.clip(-pos[0] * 0.1, -self.MAX_VH, self.MAX_VH))
            des_vy = float(np.clip(-pos[1] * 0.1, -self.MAX_VH, self.MAX_VH))
            move_x = des_vx - vel[0]
            move_y = des_vy - vel[1]

        elif self.state == ControlState.LANDING:
            if h < 15.0 and dist_h < 5.0:
                target_vz = -1.0  # constant descent to break hover
            else:
                target_vz = -float(np.clip(0.1 * h, 0.4, 3.0))
            target_yaw = 0.0
            des_vx = float(np.clip(-pos[0] * 0.2, -5, 5))
            des_vy = float(np.clip(-pos[1] * 0.2, -5, 5))
            move_x = des_vx - vel[0]
            move_y = des_vy - vel[1]

        # Deadband on horizontal commands
        if abs(move_x) < 0.05:
            move_x = 0.0
        if abs(move_y) < 0.05:
            move_y = 0.0

        # ---- Horizontal PID → body frame ----
        wx = float(np.clip(self.vx_pid.update(move_x), -1, 1))
        wy = float(np.clip(self.vy_pid.update(move_y), -1, 1))

        # World→body coordinate transform
        cy = math.cos(-yaw)
        sy = math.sin(-yaw)
        bx = wx * cy - wy * sy
        by = wx * sy + wy * cy

        # ---- Authority separation ----
        if self.state == ControlState.LANDING:
            # Terminal: flat attitude + precision RCS
            pitch_tilt = 0.0
            roll_tilt = 0.0
            bx_rcs, by_rcs = bx, by
        elif self.state == ControlState.ORIENT:
            pitch_tilt = 0.0
            roll_tilt = 0.0
            bx_rcs = bx * 0.5
            by_rcs = by * 0.5
        else:
            # TRANSIT
            if dist_h > 50.0:
                # Cruise: tilt-only
                pitch_tilt = float(np.clip(bx * 0.5, -0.35, 0.35))
                roll_tilt = float(np.clip(-by * 0.5, -0.35, 0.35))
                bx_rcs = 0.0
                by_rcs = 0.0
            else:
                # Approach: tilt + RCS
                pitch_tilt = float(np.clip(bx * 0.5, -0.5, 0.5))
                roll_tilt = float(np.clip(-by * 0.5, -0.5, 0.5))
                bx_rcs = bx
                by_rcs = by

        # Rate-limited target smoothing
        MAX_TILT_RATE = math.radians(0.1)  # ultra-slow for floating feel
        delta_p = clamp(pitch_tilt - self.target_pitch_smooth, -MAX_TILT_RATE, MAX_TILT_RATE)
        self.target_pitch_smooth += delta_p
        delta_r = clamp(roll_tilt - self.target_roll_smooth, -MAX_TILT_RATE, MAX_TILT_RATE)
        self.target_roll_smooth += delta_r

        # Attitude error → PID
        p_err = wrap_to_pi(self.target_pitch_smooth - pitch)
        r_err = wrap_to_pi(self.target_roll_smooth - r)

        p_cmd = float(np.clip(self.pitch_pid.update(p_err), -1, 1))
        r_cmd = float(np.clip(self.roll_pid.update(r_err), -1, 1))
        y_err = wrap_to_pi(target_yaw - yaw)
        y_cmd = float(np.clip(self.yaw_pid.update(y_err, is_angular=True), -1, 1))

        # Vertical PID
        vz_cmd = float(np.clip(self.vz_pid.update(target_vz - vz), -1.0, 1.0))

        # Angular comfort cap
        if self.state != ControlState.RECOVERY:
            w_mag = float(np.linalg.norm(ang_vel))
            if w_mag > self.MAX_W:
                scale = self.MAX_W / w_mag
                p_cmd *= scale
                r_cmd *= scale
                y_cmd *= scale

        # Command smoothing (LPF)
        alpha_cmd = 0.2
        self.p_cmd_smooth = alpha_cmd * p_cmd + (1 - alpha_cmd) * self.p_cmd_smooth
        self.r_cmd_smooth = alpha_cmd * r_cmd + (1 - alpha_cmd) * self.r_cmd_smooth
        self.y_cmd_smooth = alpha_cmd * y_cmd + (1 - alpha_cmd) * self.y_cmd_smooth

        p_cmd = self.p_cmd_smooth
        r_cmd = self.r_cmd_smooth
        y_cmd = self.y_cmd_smooth

        # Deadband on commands
        if abs(p_cmd) < 0.05:
            p_cmd = 0.0
        if abs(r_cmd) < 0.05:
            r_cmd = 0.0
        if abs(y_cmd) < 0.05:
            y_cmd = 0.0

        # ---- Thrust with slew rate limiting ----
        cos_safe = max(0.1, math.cos(pitch) * math.cos(r))
        target_thrust_raw = (self.gravity_ff + vz_cmd) / cos_safe
        target_thrust = float(np.clip(target_thrust_raw, 0.0, 1.0))

        if self.state != ControlState.RECOVERY:
            thrust_delta = clamp(target_thrust - self.prev_thrust,
                                 -self.THRUST_SLEW, self.THRUST_SLEW)
            final_thrust = self.prev_thrust + thrust_delta
        else:
            final_thrust = target_thrust

        # Hard engine cutoff on contact during landing
        if any_contact and self.state == ControlState.LANDING:
            final_thrust = 0.0

        self.prev_thrust = final_thrust

        # ---- RCS smoothing ----
        alpha_rcs = 0.2
        self.bx_rcs_smooth = alpha_rcs * bx_rcs + (1 - alpha_rcs) * self.bx_rcs_smooth
        self.by_rcs_smooth = alpha_rcs * by_rcs + (1 - alpha_rcs) * self.by_rcs_smooth

        cx = float(np.clip(self.bx_rcs_smooth, -1, 1))
        cy_rcs = float(np.clip(self.by_rcs_smooth, -1, 1))
        if abs(cx) < 0.05:
            cx = 0.0
        if abs(cy_rcs) < 0.05:
            cy_rcs = 0.0

        # Combine attitude + translation for tilt axes
        u_roll = float(np.clip(r_cmd + cx, -1, 1))
        u_pitch = float(np.clip(p_cmd + cy_rcs, -1, 1))
        u_yaw = float(np.clip(y_cmd, -1, 1))

        # ---- Build action ----
        action = np.zeros(16, dtype=np.float32)
        action[0] = float(final_thrust)
        action[1] = 0.0  # gimbal pitch
        action[2] = 0.0  # gimbal roll

        rcs = np.zeros(12, dtype=float)
        rcs[0] = clamp(max(0.0, u_roll), 0.0, 1.0)
        rcs[1] = clamp(max(0.0, -u_roll), 0.0, 1.0)
        rcs[2] = clamp(max(0.0, u_pitch), 0.0, 1.0)
        rcs[3] = clamp(max(0.0, -u_pitch), 0.0, 1.0)
        rcs[8] = clamp(max(0.0, u_yaw), 0.0, 1.0)
        rcs[9] = clamp(max(0.0, -u_yaw), 0.0, 1.0)
        action[3:15] = (rcs * 2.0 - 1.0).astype(np.float32)

        legs_cmd = 1.0 if h < 6.0 else -1.0
        action[15] = float(legs_cmd)

        # Telemetry
        if self.step_count % 10 == 0:
            print(
                f"[SM] {self.step_count:<5} | St={self.state.name:<10} | "
                f"H={h:5.1f} | Vz={vz:5.2f}/{target_vz:.1f} | "
                f"Dist={dist_h:5.1f} | Thr={final_thrust:.2f} | "
                f"C={any_contact}"
            )

        debug = {
            "state": float(self.state.value),
            "dist_h": dist_h,
            "alt": h,
            "target_vz": float(target_vz),
            "target_yaw": math.degrees(float(target_yaw)),
            "vz": vz,
            "r_deg": math.degrees(r),
            "p_deg": math.degrees(pitch),
            "yaw_deg": math.degrees(yaw),
            "fuel": fuel,
            "g_force": self.g_force,
            "any_contact": 1.0 if any_contact else 0.0,
        }
        return action, debug
