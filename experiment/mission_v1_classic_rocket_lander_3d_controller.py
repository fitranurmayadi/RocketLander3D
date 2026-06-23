"""Mission V1 (Classic) controller – LunarLander3D-aligned rewrite.

State machine:
  RECOVERY → ORIENT → APPROACH → ALIGN_NORTH → LANDING

Ported features from reference LunarLander3D mission_v1_classic:
  - gravity feedforward
  - G-force estimation
  - coordinate transform world→body for horizontal RCS
  - command smoothing (low-pass filter)
  - altitude profile descent with progress tracking
  - start_dist_h for smooth approach
"""

import math
from enum import Enum
from typing import Dict, Tuple

import numpy as np
import pybullet as pb


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def wrap_to_pi(x: float) -> float:
    return (x + math.pi) % (2 * math.pi) - math.pi


def quat_to_euler(quat_xyzw) -> Tuple[float, float, float]:
    r, pitch, yaw = pb.getEulerFromQuaternion(quat_xyzw)
    return float(r), float(pitch), float(yaw)


# ---------------------------------------------------------------------------
# Mission phases
# ---------------------------------------------------------------------------

class ControlState(Enum):
    RECOVERY = 1      # Recover from random tumble
    ORIENT = 2        # Yaw toward target before moving
    APPROACH = 3      # Fly to waypoint & descend
    ALIGN_NORTH = 4   # Yaw to 0.0
    LANDING = 5       # Terminal descent


# ---------------------------------------------------------------------------
# PID controller
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

class MissionV1ClassicRocketLander3DController:
    """Classic 5-phase state-machine controller (LunarLander3D V1 style)."""

    def __init__(self, dt: float):
        self.dt = float(dt)

        # Gravity feedforward for RocketLander env (g=9.81, MAIN_ENGINE_POWER=60000, mass=1000)
        # Hover throttle ≈ mass*g / MAIN_ENGINE_POWER = 1000*9.81/60000 ≈ 0.1635
        # Action throttle is mapped as (action[0]+1)/2, so action[0] for hover ≈ 2*0.1635 - 1 ≈ -0.673
        # We work in raw throttle (0..1) and convert at the end. gravity_ff ~ 0.30 (tuned).
        self.gravity_ff = 0.30

        # Attitude PIDs
        self.roll_pid = PID(kp=2.0, ki=0.0, kd=6.0, dt=dt, limit=1.0)
        self.pitch_pid = PID(kp=2.0, ki=0.0, kd=6.0, dt=dt, limit=1.0)
        self.yaw_pid = PID(kp=0.5, ki=0.01, kd=1.0, dt=dt, limit=1.0)

        # Vertical velocity PID
        self.vz_pid = PID(kp=10.0, ki=0.1, kd=1.5, dt=dt, limit=1.0)

        # Horizontal velocity PIDs
        self.vx_pid = PID(kp=1.0, ki=0.01, kd=0.5, dt=dt, limit=1.0)
        self.vy_pid = PID(kp=1.0, ki=0.01, kd=0.5, dt=dt, limit=1.0)

        # Mission thresholds (scaled for RocketLander env)
        self.WAYPOINT_ALT = 50.0
        self.WAYPOINT_DIST = 8.0
        self.MAX_SAFE_TILT_DEG = 45.0
        self.ATT_GOOD_DEG = 10.0

        # State tracking
        self.state = ControlState.RECOVERY
        self.step_count = 0
        self.attitude_good_count = 0
        self.last_valid_target_yaw = 0.0
        self.start_dist_h = None

        # Command smoothing
        self.move_x_smooth = 0.0
        self.move_y_smooth = 0.0

        # G-force estimation
        self.prev_vel = np.zeros(3)
        self.g_force = 1.0

        self.reset()

    def reset(self):
        self.state = ControlState.RECOVERY
        self.step_count = 0
        self.attitude_good_count = 0
        self.last_valid_target_yaw = 0.0
        self.start_dist_h = None
        self.move_x_smooth = 0.0
        self.move_y_smooth = 0.0
        self.prev_vel = np.zeros(3)
        self.g_force = 1.0
        for pid in [self.roll_pid, self.pitch_pid, self.yaw_pid,
                    self.vz_pid, self.vx_pid, self.vy_pid]:
            pid.reset()

    def _reset_horizontal_pids(self):
        self.vx_pid.reset()
        self.vy_pid.reset()
        self.move_x_smooth = 0.0
        self.move_y_smooth = 0.0

    # ----- state machine transitions -----
    def _update_state_machine(self, r_deg: float, p_deg: float, yaw: float,
                              h: float, vz: float, dist_h: float):
        is_failing = abs(r_deg) > self.MAX_SAFE_TILT_DEG or abs(p_deg) > self.MAX_SAFE_TILT_DEG
        is_stable = abs(r_deg) < self.ATT_GOOD_DEG and abs(p_deg) < self.ATT_GOOD_DEG

        if is_stable:
            self.attitude_good_count += 1
        else:
            self.attitude_good_count = 0

        attitude_ok = self.attitude_good_count > 20
        waypoint_reached = dist_h < self.WAYPOINT_DIST and abs(h - self.WAYPOINT_ALT) < 20.0

        yaw_to_target_ok = abs(math.degrees(wrap_to_pi(self.last_valid_target_yaw - yaw))) < 10.0
        north_aligned = abs(math.degrees(wrap_to_pi(0.0 - yaw))) < 15.0

        if self.state == ControlState.RECOVERY:
            if attitude_ok:
                print("[SM] >> RECOVERY COMPLETE. Orienting toward target.")
                self.state = ControlState.ORIENT
                self._reset_horizontal_pids()
                self.attitude_good_count = 0

        elif self.state == ControlState.ORIENT:
            if is_failing:
                self.state = ControlState.RECOVERY
            elif yaw_to_target_ok:
                print("[SM] >> ORIENTATION COMPLETE. Fly to Waypoint.")
                self.state = ControlState.APPROACH
                self._reset_horizontal_pids()

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

    # ----- main act -----
    def act(self, obs: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        self.step_count += 1

        # ---- Parse observation (RocketLander-v0, normalize_obs=True) ----
        pos = obs[0:3] * 500.0
        quat = obs[3:7]
        vel = obs[7:10] * 50.0
        ang_vel_local = obs[10:13] * 10.0
        contacts = obs[13:17]
        h = float(obs[17] * 500.0)
        fuel = float(obs[18])

        r, pitch, yaw = quat_to_euler(quat)
        dist_h = float(np.linalg.norm(pos[0:2]))
        vz = float(vel[2])

        # Yaw target: face pad (latch when near to avoid atan2 noise)
        if self.state in [ControlState.ORIENT, ControlState.APPROACH]:
            if dist_h > 5.0:
                self.last_valid_target_yaw = math.atan2(-pos[1], -pos[0])
            target_yaw_rad = self.last_valid_target_yaw
        elif self.state == ControlState.RECOVERY:
            target_yaw_rad = yaw  # maintain current
        else:
            target_yaw_rad = 0.0  # align north

        # State machine update
        self._update_state_machine(
            r_deg=math.degrees(r), p_deg=math.degrees(pitch),
            yaw=yaw, h=h, vz=vz, dist_h=dist_h
        )

        # ---- G-force estimation ----
        accel = (vel - self.prev_vel) / max(1e-9, self.dt)
        gravity_vec = np.array([0.0, 0.0, -9.81])
        self.g_force = float(np.linalg.norm(accel - gravity_vec) / 9.81)
        self.prev_vel = vel.copy()

        # ---- Control targets per state ----
        target_vz = 0.0
        target_yaw = 0.0
        move_x, move_y = 0.0, 0.0
        target_h = h  # default: hold

        if self.state == ControlState.RECOVERY:
            target_vz = 0.0
            target_yaw = yaw
            move_x = -vel[0]
            move_y = -vel[1]

        elif self.state == ControlState.ORIENT:
            target_vz = 0.0
            target_yaw = target_yaw_rad
            move_x = -vel[0]
            move_y = -vel[1]

        elif self.state == ControlState.APPROACH:
            # Progress-based altitude profile (reference style)
            if self.start_dist_h is None:
                self.start_dist_h = dist_h
            progress = float(np.clip(
                (self.start_dist_h - dist_h) / max(1.0, self.start_dist_h), 0.0, 1.0
            ))
            target_h = h + (self.WAYPOINT_ALT - h) * progress  # blend toward waypoint alt
            alt_err = target_h - h
            target_vz = float(np.clip(alt_err * 0.1, -10.0, 5.0))

            target_yaw = target_yaw_rad

            # Horizontal: fly toward origin
            if dist_h > 1.0:
                dir_x = -pos[0] / dist_h
                dir_y = -pos[1] / dist_h
                speed = float(np.clip(dist_h * 0.1, 2.0, 10.0))
                des_vx = dir_x * speed
                des_vy = dir_y * speed
            else:
                des_vx = des_vy = 0.0
            move_x = des_vx - vel[0]
            move_y = des_vy - vel[1]

        elif self.state == ControlState.ALIGN_NORTH:
            target_vz = float(np.clip((self.WAYPOINT_ALT - h) * 0.1, -5.0, 5.0))
            target_yaw = 0.0
            move_x = -vel[0]
            move_y = -vel[1]

        elif self.state == ControlState.LANDING:
            target_yaw = 0.0
            # Precision lateral correction
            des_vx = float(np.clip(-pos[0] * 0.1, -4, 4)) if abs(pos[0]) > 0.05 else 0.0
            des_vy = float(np.clip(-pos[1] * 0.1, -4, 4)) if abs(pos[1]) > 0.05 else 0.0
            move_x = des_vx - vel[0]
            move_y = des_vy - vel[1]
            # Staged descent (reference style)
            if h > 30:
                target_vz = -3.0
            elif h > 10:
                target_vz = -1.5
            elif h > 3:
                target_vz = -0.7
            else:
                target_vz = -0.3

        # ---- Attitude PID commands ----
        r_cmd = float(np.clip(self.roll_pid.update(-r), -1, 1))
        p_cmd = float(np.clip(self.pitch_pid.update(-pitch), -1, 1))
        y_err = wrap_to_pi(target_yaw - yaw)
        y_cmd = float(np.clip(self.yaw_pid.update(y_err, is_angular=True), -1, 1))

        # ---- Horizontal commands with smoothing ----
        alpha = 0.2
        self.move_x_smooth = alpha * move_x + (1 - alpha) * self.move_x_smooth
        self.move_y_smooth = alpha * move_y + (1 - alpha) * self.move_y_smooth

        wx = float(np.clip(self.vx_pid.update(self.move_x_smooth), -1, 1))
        wy = float(np.clip(self.vy_pid.update(self.move_y_smooth), -1, 1))

        # Coordinate transform: world → body frame (reference style)
        cy = math.cos(-yaw)
        sy = math.sin(-yaw)
        bx = wx * cy - wy * sy
        by = wx * sy + wy * cy

        # Combine attitude + horizontal for tilt axes
        u_roll = float(np.clip(r_cmd + bx, -1, 1))
        u_pitch = float(np.clip(p_cmd + by, -1, 1))
        u_yaw = float(np.clip(y_cmd, -1, 1))

        # ---- Vertical thrust ----
        vz_out = float(self.vz_pid.update(target_vz - vz))
        throttle = clamp(self.gravity_ff + vz_out, 0.0, 1.0)

        # ---- Legs deploy ----
        legs_cmd = 1.0 if h < 6.0 else -1.0

        # ---- Build action vector (RocketLander-v0: 16 dims) ----
        action = np.zeros(16, dtype=np.float32)
        action[0] = float(throttle)
        action[1] = 0.0  # gimbal pitch
        action[2] = 0.0  # gimbal roll

        # RCS mapping (placeholder – same as existing, verify with pid_authority_report)
        rcs = np.zeros(12, dtype=float)
        rcs[0] = clamp(max(0.0, u_roll), 0.0, 1.0)
        rcs[1] = clamp(max(0.0, -u_roll), 0.0, 1.0)
        rcs[2] = clamp(max(0.0, u_pitch), 0.0, 1.0)
        rcs[3] = clamp(max(0.0, -u_pitch), 0.0, 1.0)
        rcs[8] = clamp(max(0.0, u_yaw), 0.0, 1.0)
        rcs[9] = clamp(max(0.0, -u_yaw), 0.0, 1.0)
        action[3:15] = (rcs * 2.0 - 1.0).astype(np.float32)
        action[15] = float(legs_cmd)

        # ---- Telemetry ----
        if self.step_count % 50 == 0:
            print(
                f"[SM] St={self.state.name:<12} | H={h:6.1f} | "
                f"Vz={vz:5.2f}/{target_vz:4.1f} | Y={math.degrees(yaw):6.1f} | "
                f"Dist={dist_h:6.1f}"
            )

        debug = {
            "state": float(self.state.value),
            "dist_h": dist_h,
            "alt": h,
            "target_vz": float(target_vz),
            "target_yaw": math.degrees(float(target_yaw)),
            "target_h": float(target_h),
            "vz": vz,
            "r_deg": math.degrees(r),
            "p_deg": math.degrees(pitch),
            "yaw_deg": math.degrees(yaw),
            "fuel": fuel,
            "g_force": self.g_force,
        }
        return action, debug
