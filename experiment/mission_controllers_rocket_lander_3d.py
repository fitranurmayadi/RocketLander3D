import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple

import numpy as np
import pybullet as p


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def wrap_to_pi(x: float) -> float:
    return (x + math.pi) % (2 * math.pi) - math.pi


def quat_to_euler(quat_xyzw) -> Tuple[float, float, float]:
    # env uses pybullet, quaternion ordering from obs[3:7]
    r, pitch, yaw = p.getEulerFromQuaternion(quat_xyzw)
    return float(r), float(pitch), float(yaw)


class MissionPhase(Enum):
    RECOVERY = 1
    ORIENT = 2
    TRANSIT = 3
    ALIGN_NORTH = 4
    LANDING = 5


@dataclass
class MissionGains:
    roll_pid: Tuple[float, float, float]
    pitch_pid: Tuple[float, float, float]
    yaw_pid: Tuple[float, float, float]
    vx_pid: Tuple[float, float, float]
    vy_pid: Tuple[float, float, float]
    vz_pid: Tuple[float, float, float]


class PID:
    def __init__(self, kp: float, ki: float, kd: float, dt: float, limit: float = 1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.limit = float(limit)
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
        self.integral = float(np.clip(self.integral, -self.limit, self.limit))

        diff = error - self.prev_error
        derivative = diff / max(1e-9, self.dt)
        self.prev_error = error

        out = self.kp * error + self.ki * self.integral + self.kd * derivative
        return float(np.clip(out, -self.limit, self.limit))


class MissionBase:
    def __init__(self, dt: float = 1.0 / 60.0):
        self.dt = float(dt)
        self.state: MissionPhase = MissionPhase.RECOVERY
        self.step_count = 0
        self.prev_thrust = 0.0
        self.prev_vel = np.zeros(3, dtype=float)

        # smoothed targets/commands
        self.target_roll = 0.0
        self.target_pitch = 0.0
        self.target_yaw = 0.0

        self.roll_pid: PID
        self.pitch_pid: PID
        self.yaw_pid: PID
        self.vx_pid: PID
        self.vy_pid: PID
        self.vz_pid: PID

        self.reset()

    def reset(self):
        self.state = MissionPhase.RECOVERY
        self.step_count = 0
        self.prev_thrust = 0.0
        self.prev_vel[:] = 0.0
        self.target_roll = 0.0
        self.target_pitch = 0.0
        self.target_yaw = 0.0

    def _extract_obs(self, obs: np.ndarray) -> Dict[str, float]:
        # rocket_lander_env observation is normalized by VecNormalize but can be raw.
        # In mission scripts we typically run env(normalize_obs=True) and rely on obs being normalized.
        # Here we assume mission scripts pass obs already denormalized or use env(normalize_obs=True) but controller uses raw scales.
        raise NotImplementedError

    def act(self, obs: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        raise NotImplementedError


class ClassicStateMachineRocket(MissionBase):
    """mission_v1_classic_rocket_lander_3d

    Mirrors LunarLander3D mission_v1_classic structure:
    - RECOVERY: recover attitude (level + damp)
    - ORIENT: yaw/orient to target
    - APPROACH/TRANSIT: go to waypoint height
    - ALIGN_NORTH: yaw=0
    - LANDING: terminal descent

    This implementation is a *first port*: you MUST tune gains + RCS mixing signs.
    """

    def __init__(self, dt: float = 1.0 / 60.0):
        super().__init__(dt=dt)

        # These are placeholders. Tune using your calibration scripts (pid_authority_report / verify_*.py).
        gains = MissionGains(
            roll_pid=(20.0, 0.0, 6.0),
            pitch_pid=(20.0, 0.0, 6.0),
            yaw_pid=(5.0, 0.01, 1.0),
            vx_pid=(10.0, 0.01, 0.5),
            vy_pid=(10.0, 0.01, 0.5),
            vz_pid=(10.0, 0.1, 1.5),
        )

        self.roll_pid = PID(*gains.roll_pid, dt=self.dt, limit=1.0)
        self.pitch_pid = PID(*gains.pitch_pid, dt=self.dt, limit=1.0)
        self.yaw_pid = PID(*gains.yaw_pid, dt=self.dt, limit=1.0)
        self.vx_pid = PID(*gains.vx_pid, dt=self.dt, limit=1.0)
        self.vy_pid = PID(*gains.vy_pid, dt=self.dt, limit=1.0)
        self.vz_pid = PID(*gains.vz_pid, dt=self.dt, limit=1.0)

        # Mission params (adapted from reference, but scaled for Rocket env)
        self.WAYPOINT_ALT = 50.0
        self.WAYPOINT_DIST = 8.0

        self.MAX_SAFE_TILT_DEG = 90.0
        self.ATT_GOOD_DEG = 10.0

        self.last_valid_target_yaw = 0.0

        # For thrust feedforward: env gravity is ~-9.81, and action[0] throttle maps to force magnitude.
        # In rocket_lander_env, thrust is MAIN_ENGINE_POWER * throttle_cmd and then applied as external force.
        # So we cannot directly port Lunar's 0.368 gravity_ff. We'll keep throttle based on vz_pid only.
        self.gravity_ff = 0.98

        self.reset()

    def reset_horizontal_pids(self):
        self.vx_pid.reset()
        self.vy_pid.reset()

    def update_state_machine(self, r: float, p_ang: float, vz: float, h: float, yaw: float, dist_h: float):
        # thresholds in degrees for attitude
        r_deg = math.degrees(r)
        p_deg = math.degrees(p_ang)

        is_failing = abs(r_deg) > self.MAX_SAFE_TILT_DEG or abs(p_deg) > self.MAX_SAFE_TILT_DEG
        is_stable_now = abs(r_deg) < self.ATT_GOOD_DEG and abs(p_deg) < self.ATT_GOOD_DEG

        if is_stable_now:
            self.attitude_good_count = getattr(self, "attitude_good_count", 0) + 1
        else:
            self.attitude_good_count = 0

        attitude_ok = self.attitude_good_count > 20
        waypoint_reached = dist_h < self.WAYPOINT_DIST and abs(h - self.WAYPOINT_ALT) < 20.0

        yaw_err = wrap_to_pi(0.0 - yaw)
        north_aligned = abs(math.degrees(yaw_err)) < 15.0

        # Target yaw logic: in ORIENT/TRANSIT we'll point toward pad (computed in mission script)
        # Here we only use yaw=0 as the ALIGN target.

        if self.state == MissionPhase.RECOVERY:
            if attitude_ok:
                self.state = MissionPhase.ORIENT
                self.reset_horizontal_pids()
                self.attitude_good_count = 0

        elif self.state == MissionPhase.ORIENT:
            if is_failing:
                self.state = MissionPhase.RECOVERY
            elif abs(math.degrees(wrap_to_pi(self.last_valid_target_yaw - yaw))) < 10.0:
                self.state = MissionPhase.TRANSIT
                self.reset_horizontal_pids()

        elif self.state == MissionPhase.TRANSIT:
            if is_failing:
                self.state = MissionPhase.RECOVERY
            elif waypoint_reached:
                self.state = MissionPhase.ALIGN_NORTH

        elif self.state == MissionPhase.ALIGN_NORTH:
            if is_failing:
                self.state = MissionPhase.RECOVERY
            elif north_aligned:
                self.state = MissionPhase.LANDING

        elif self.state == MissionPhase.LANDING:
            if is_failing:
                self.state = MissionPhase.RECOVERY

    def act(self, obs: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        self.step_count += 1

        # rocket_lander_env obs normalized (normalize_obs=True). Use scales from env: POS_SCALE=500, VEL_SCALE=50, ANG_VEL_SCALE=10, ALT_SCALE=500.
        # We'll assume mission is run with normalize_obs=True.
        pos = obs[0:3] * 500.0
        quat = obs[3:7]
        vel = obs[7:10] * 50.0
        ang_vel_local = obs[10:13] * 10.0
        foot_contacts = obs[13:17]  # already 0/1
        alt = float(obs[17] * 500.0)
        fuel = float(obs[18])

        r, pitch, yaw = quat_to_euler(quat)
        wx, wy, wz = ang_vel_local

        dist_h = float(np.linalg.norm(pos[0:2]))
        vz = float(vel[2])
        h = alt

        # Yaw target: face pad
        if dist_h > 1e-3:
            self.last_valid_target_yaw = math.atan2(-pos[1], -pos[0])

        self.update_state_machine(r, pitch, vz, h, yaw, dist_h)

        # --- Targets ---
        target_vz = 0.0
        target_yaw = 0.0
        move_x = 0.0
        move_y = 0.0

        if self.state == MissionPhase.RECOVERY:
            target_vz = 0.0
            target_yaw = yaw
            move_x = -vel[0]
            move_y = -vel[1]
        elif self.state == MissionPhase.ORIENT:
            target_vz = 0.0
            target_yaw = self.last_valid_target_yaw
            move_x = -vel[0]
            move_y = -vel[1]
        elif self.state == MissionPhase.TRANSIT:
            target_vz = -5.0 if h > 70.0 else -2.0
            target_yaw = self.last_valid_target_yaw
            move_x = float(np.clip(-pos[0] * 0.05, -8, 8) - vel[0])
            move_y = float(np.clip(-pos[1] * 0.05, -8, 8) - vel[1])
        elif self.state == MissionPhase.ALIGN_NORTH:
            target_vz = -2.5
            target_yaw = 0.0
            move_x = -vel[0]
            move_y = -vel[1]
        else:  # LANDING
            target_yaw = 0.0
            move_x = float(np.clip(-pos[0] * 0.1, -4, 4) - vel[0])
            move_y = float(np.clip(-pos[1] * 0.1, -4, 4) - vel[1])
            if h > 20.0:
                target_vz = -1.5
            else:
                target_vz = -0.7

        # --- Outer attitude commands -> RCS (placeholder mapping) ---
        # Use attitude PIDs on (target angle - actual angle)
        r_cmd = self.roll_pid.update(-r)
        p_cmd = self.pitch_pid.update(-pitch)
        y_err = wrap_to_pi(target_yaw - yaw)
        y_cmd = self.yaw_pid.update(y_err, is_angular=True)

        # --- Translational commands mapped to roll/pitch via vx/vy pid ---
        # First compute desired body rates proxies from move_x/move_y
        wx_cmd = float(np.clip(self.vx_pid.update(move_x), -1, 1))
        wy_cmd = float(np.clip(self.vy_pid.update(move_y), -1, 1))

        # Combine attitude and translational for tilt axes (simple add; tune later)
        u_roll = float(np.clip(r_cmd + wx_cmd, -1, 1))
        u_pitch = float(np.clip(p_cmd + wy_cmd, -1, 1))
        u_yaw = float(np.clip(y_cmd, -1, 1))

        # Vertical thrust: use vz pid
        vz_out = float(self.vz_pid.update(target_vz - vz))
        throttle = clamp(self.gravity_ff + vz_out, 0.0, 1.0)

        # Legs deploy
        legs_cmd = 1.0 if alt < 6.0 else -1.0

        # --- Mix into rocket_lander_env action ---
        # rocket_lander_env action:
        # [0]=throttle, [1]=gimbal pitch, [2]=gimbal roll, [3:15]=RCS 12, [15]=legs
        action = np.zeros(16, dtype=np.float32)
        action[0] = throttle
        action[1] = 0.0
        action[2] = 0.0

        # Placeholder RCS mixing (ABS-only). Replace after verify scripts.
        rcs = np.zeros(12, dtype=float)
        # roll: indices 0..3 are pr; yaw indices 8..11 (see rocket_lander_env)
        rcs[0] = clamp(max(0.0, u_roll), 0.0, 1.0)
        rcs[1] = clamp(max(0.0, -u_roll), 0.0, 1.0)

        # pitch: indices 2..3
        rcs[2] = clamp(max(0.0, u_pitch), 0.0, 1.0)
        rcs[3] = clamp(max(0.0, -u_pitch), 0.0, 1.0)

        # yaw: indices 8..9 (partial; placeholder)
        rcs[8] = clamp(max(0.0, u_yaw), 0.0, 1.0)
        rcs[9] = clamp(max(0.0, -u_yaw), 0.0, 1.0)

        action[3:15] = (rcs * 2.0 - 1.0).astype(np.float32)
        action[15] = float(legs_cmd)

        debug = {
            "state": float(self.state.value),
            "dist_h": dist_h,
            "alt": h,
            "target_vz": target_vz,
            "target_yaw": math.degrees(target_yaw),
            "vz": vz,
            "r_deg": math.degrees(r),
            "p_deg": math.degrees(pitch),
            "yaw_deg": math.degrees(yaw),
            "fuel": fuel,
        }
        return action, debug


class DirectRocketLanderStateMachineRocket(MissionBase):
    """mission_v2_direct_rocket_lander_3d

    Port skeleton based on Lunar mission_v2_direct:
    - Similar state machine
    - authority separation for TRANSIT/LANDING
    - thrust uses pitch/roll cos-safe approach is not present in rocket env,
      so we keep a simpler vz controller.

    Needs tuning + correct RCS mapping signs.
    """

    def __init__(self, dt: float = 1.0 / 60.0):
        super().__init__(dt=dt)

        gains = MissionGains(
            roll_pid=(1.2, 0.0, 2.5),
            pitch_pid=(1.2, 0.0, 2.5),
            yaw_pid=(0.6, 0.01, 1.2),
            vx_pid=(0.2, 0.0, 0.2),
            vy_pid=(0.2, 0.0, 0.2),
            vz_pid=(8.0, 0.15, 1.2),
        )
        self.roll_pid = PID(*gains.roll_pid, dt=self.dt, limit=1.0)
        self.pitch_pid = PID(*gains.pitch_pid, dt=self.dt, limit=1.0)
        self.yaw_pid = PID(*gains.yaw_pid, dt=self.dt, limit=1.0)
        self.vx_pid = PID(*gains.vx_pid, dt=self.dt, limit=1.0)
        self.vy_pid = PID(*gains.vy_pid, dt=self.dt, limit=1.0)
        self.vz_pid = PID(*gains.vz_pid, dt=self.dt, limit=1.0)

        self.WAYPOINT_ALT = 50.0
        self.WAYPOINT_DIST = 8.0
        self.MAX_SAFE_TILT_DEG = 30.0
        self.MAX_W_HZ = math.radians(30.0)

        self.last_valid_target_yaw = 0.0
        self.reset()

    def reset_horizontal_pids(self):
        self.vx_pid.reset()
        self.vy_pid.reset()

    def update_state_machine(self, r: float, p_ang: float, vz: float, h: float, yaw: float, dist_h: float, any_contact: bool):
        r_deg = abs(math.degrees(r))
        p_deg = abs(math.degrees(p_ang))
        is_failing = r_deg > self.MAX_SAFE_TILT_DEG or p_deg > self.MAX_SAFE_TILT_DEG

        if self.state == MissionPhase.RECOVERY:
            if r_deg < 10.0 and p_deg < 10.0 and abs(vz) < 6.0:
                self.state = MissionPhase.ORIENT
                self.reset_horizontal_pids()

        elif self.state == MissionPhase.ORIENT:
            y_err = abs(math.degrees(wrap_to_pi(self.last_valid_target_yaw - yaw)))
            if is_failing:
                self.state = MissionPhase.RECOVERY
            elif y_err < 8.0:
                self.state = MissionPhase.TRANSIT
                self.reset_horizontal_pids()

        elif self.state == MissionPhase.TRANSIT:
            if is_failing:
                self.state = MissionPhase.RECOVERY
            elif any_contact and h < 40.0:
                self.state = MissionPhase.LANDING
                self.reset_horizontal_pids()
            elif dist_h < 5.0 and h < 80.0:
                self.state = MissionPhase.LANDING

        elif self.state == MissionPhase.LANDING:
            if is_failing:
                self.state = MissionPhase.RECOVERY
            elif dist_h > 25.0:
                self.state = MissionPhase.TRANSIT
                self.reset_horizontal_pids()

    def act(self, obs: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        self.step_count += 1

        pos = obs[0:3] * 500.0
        quat = obs[3:7]
        vel = obs[7:10] * 50.0
        ang_vel_local = obs[10:13] * 10.0
        contacts = obs[13:17]
        alt = float(obs[17] * 500.0)
        fuel = float(obs[18])

        r, pitch, yaw = quat_to_euler(quat)
        wx, wy, wz = ang_vel_local

        dist_h = float(np.linalg.norm(pos[0:2]))
        h = alt
        vz = float(vel[2])

        if dist_h > 1e-3:
            self.last_valid_target_yaw = math.atan2(-pos[1], -pos[0])

        any_contact = bool(np.max(contacts) > 0.5) and h < 5.0

        self.update_state_machine(r, pitch, vz, h, yaw, dist_h, any_contact)

        # targets
        if self.state == MissionPhase.RECOVERY:
            target_vz = 0.0
            target_yaw = yaw
            move_x, move_y = -vel[0], -vel[1]
        elif self.state == MissionPhase.ORIENT:
            target_vz = 0.0
            target_yaw = self.last_valid_target_yaw
            move_x, move_y = -vel[0], -vel[1]
        elif self.state == MissionPhase.TRANSIT:
            target_vz = -5.0 if h > 70.0 else -2.0
            target_yaw = self.last_valid_target_yaw
            move_x = float(np.clip(-pos[0] * 0.03, -8, 8) - vel[0])
            move_y = float(np.clip(-pos[1] * 0.03, -8, 8) - vel[1])
        else:
            # landing
            target_yaw = 0.0
            move_x = float(np.clip(-pos[0] * 0.08, -5, 5) - vel[0])
            move_y = float(np.clip(-pos[1] * 0.08, -5, 5) - vel[1])
            target_vz = -1.0 if h < 15.0 and dist_h < 5.0 else -np.clip(0.08 * h, 0.4, 3.0)

        # authority separation:
        # during TRANSIT, keep smaller attitude commands when far
        far = dist_h > 50.0
        if self.state == MissionPhase.TRANSIT and far:
            tilt_scale = 0.5
        elif self.state == MissionPhase.TRANSIT:
            tilt_scale = 1.0
        elif self.state == MissionPhase.LANDING:
            tilt_scale = 1.0
        else:
            tilt_scale = 0.6

        r_cmd = self.roll_pid.update(-r) * tilt_scale
        p_cmd = self.pitch_pid.update(-pitch) * tilt_scale
        y_err = wrap_to_pi(target_yaw - yaw)
        y_cmd = self.yaw_pid.update(y_err, is_angular=True) * tilt_scale

        wx_cmd = float(np.clip(self.vx_pid.update(move_x), -1, 1)) * tilt_scale
        wy_cmd = float(np.clip(self.vy_pid.update(move_y), -1, 1)) * tilt_scale

        u_roll = float(np.clip(r_cmd + wx_cmd, -1, 1))
        u_pitch = float(np.clip(p_cmd + wy_cmd, -1, 1))
        u_yaw = float(np.clip(y_cmd, -1, 1))

        vz_out = float(self.vz_pid.update(target_vz - vz))
        throttle = clamp(0.3 + vz_out, 0.0, 1.0)

        legs_cmd = 1.0 if alt < 6.0 else -1.0

        action = np.zeros(16, dtype=np.float32)
        action[0] = throttle
        action[1] = 0.0
        action[2] = 0.0

        rcs = np.zeros(12, dtype=float)
        # Same placeholder mixing as classic
        rcs[0] = clamp(max(0.0, u_roll), 0.0, 1.0)
        rcs[1] = clamp(max(0.0, -u_roll), 0.0, 1.0)
        rcs[2] = clamp(max(0.0, u_pitch), 0.0, 1.0)
        rcs[3] = clamp(max(0.0, -u_pitch), 0.0, 1.0)
        rcs[8] = clamp(max(0.0, u_yaw), 0.0, 1.0)
        rcs[9] = clamp(max(0.0, -u_yaw), 0.0, 1.0)

        action[3:15] = (rcs * 2.0 - 1.0).astype(np.float32)
        action[15] = float(legs_cmd)

        debug = {
            "state": float(self.state.value),
            "dist_h": dist_h,
            "alt": h,
            "target_vz": target_vz,
            "target_yaw": math.degrees(target_yaw),
            "vz": vz,
            "r_deg": math.degrees(r),
            "p_deg": math.degrees(pitch),
            "yaw_deg": math.degrees(yaw),
            "fuel": fuel,
            "any_contact": 1.0 if any_contact else 0.0,
        }
        return action, debug


class TrajectoryRocketLander3D(MissionBase):
    """mission_v3_trajectory_rocket_lander_3d

    Port skeleton from Lunar mission_v3_trajectory:
    - Phase 1: pre-flight stabilization
    - Phase 2: generate 3D quintic trajectory from current state to target
    - Inner loop: position->accel -> attitude commands -> rate/attitude PID -> RCS

    Here we simplify: convert trajectory position error into desired horizontal/vertical velocity,
    then reuse attitude->RCS mixing.
    """

    def __init__(self, dt: float = 1.0 / 60.0):
        super().__init__(dt=dt)

        gains = MissionGains(
            roll_pid=(1.5, 0.0, 4.0),
            pitch_pid=(1.5, 0.0, 4.0),
            yaw_pid=(0.8, 0.01, 1.0),
            vx_pid=(0.5, 0.0, 0.2),
            vy_pid=(0.5, 0.0, 0.2),
            vz_pid=(6.0, 0.1, 2.0),
        )
        self.roll_pid = PID(*gains.roll_pid, dt=self.dt, limit=1.0)
        self.pitch_pid = PID(*gains.pitch_pid, dt=self.dt, limit=1.0)
        self.yaw_pid = PID(*gains.yaw_pid, dt=self.dt, limit=1.0)
        self.vx_pid = PID(*gains.vx_pid, dt=self.dt, limit=1.0)
        self.vy_pid = PID(*gains.vy_pid, dt=self.dt, limit=1.0)
        self.vz_pid = PID(*gains.vz_pid, dt=self.dt, limit=1.0)

        # Trajectory params
        self.trajectory_duration = 60.0
        self.target_alt = 9.9

        self._traj_start_pos = None
        self._traj_start_vel = None
        self._traj_t = 0.0

        self._mission_phase = 1
        self._stabilized = False

    def reset(self):
        super().reset()
        self._traj_start_pos = None
        self._traj_start_vel = None
        self._traj_t = 0.0
        self._mission_phase = 1
        self._stabilized = False

    def _quintic(self, xi, vi, ai, xf, vf, af, t, T):
        # Return x, v, a
        a0 = xi
        a1 = vi
        a2 = 0.5 * ai
        A = np.array([
            [T**3, T**4, T**5],
            [3*T**2, 4*T**3, 5*T**4],
            [6*T, 12*T**2, 20*T**3]
        ], dtype=float)
        b = np.array([
            xf - a0 - a1*T - a2*T**2,
            vf - a1 - 2*a2*T,
            af - 2*a2
        ], dtype=float)
        try:
            x = np.linalg.solve(A, b)
            a3, a4, a5 = x
        except np.linalg.LinAlgError:
            a3, a4, a5 = 0.0, 0.0, 0.0

        tt = t
        xt = a0 + a1*tt + a2*tt**2 + a3*tt**3 + a4*tt**4 + a5*tt**5
        vt = a1 + 2*a2*tt + 3*a3*tt**2 + 4*a4*tt**3 + 5*a5*tt**4
        at = 2*a2 + 6*a3*tt + 12*a4*tt**2 + 20*a5*tt**3
        return xt, vt, at

    def _compute_ref(self, cp, cv, t) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # end at target (0,0,target_alt) with near-zero velocity
        end_p = np.array([0.0, 0.0, self.target_alt], dtype=float)
        end_v = np.array([0.0, 0.0, -0.5], dtype=float)
        end_a = np.array([0.0, 0.0, 0.0], dtype=float)

        T = self.trajectory_duration
        tt = clamp(t, 0.0, T)

        ref_p = np.zeros(3, dtype=float)
        ref_v = np.zeros(3, dtype=float)
        ref_a = np.zeros(3, dtype=float)
        for i in range(3):
            xi, vi, ai = cp[i], cv[i], 0.0
            xf, vf, af = end_p[i], end_v[i], end_a[i]
            xt, vt, at = self._quintic(xi, vi, ai, xf, vf, af, tt, T)
            ref_p[i], ref_v[i], ref_a[i] = xt, vt, at
        return ref_p, ref_v, ref_a

    def act(self, obs: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        self.step_count += 1

        pos = obs[0:3] * 500.0
        quat = obs[3:7]
        vel = obs[7:10] * 50.0
        ang_vel_local = obs[10:13] * 10.0
        contacts = obs[13:17]
        alt = float(obs[17] * 500.0)
        fuel = float(obs[18])

        r, pitch, yaw = quat_to_euler(quat)
        wx, wy, wz = ang_vel_local
        dist_h = float(np.linalg.norm(pos[0:2]))
        vz = float(vel[2])

        # phase logic: stabilize until attitude and velocity small
        if self._mission_phase == 1:
            att_ok = (abs(math.degrees(r)) < 5.0 and abs(math.degrees(pitch)) < 5.0)
            vel_ok = (np.linalg.norm(vel) < 2.0)
            if att_ok and vel_ok:
                self._stabilized = True
                self._mission_phase = 2
                self._traj_start_pos = pos.copy()
                self._traj_start_vel = vel.copy()
                self._traj_t = 0.0

        if self._mission_phase == 1:
            target_vz = 0.0
            target_yaw = yaw
            move_x, move_y = -vel[0], -vel[1]
        else:
            if self._traj_start_pos is None:
                self._traj_start_pos = pos.copy()
                self._traj_start_vel = vel.copy()
                self._traj_t = 0.0

            ref_p, ref_v, ref_a = self._compute_ref(self._traj_start_pos, self._traj_start_vel, self._traj_t)
            self._traj_t += self.dt

            # horizontal control based on tracking ref_v
            target_vz = float(ref_v[2])
            target_yaw = 0.0
            move_x = float(ref_v[0] - vel[0])
            move_y = float(ref_v[1] - vel[1])

        # attitude pid commands
        r_cmd = self.roll_pid.update(-r)
        p_cmd = self.pitch_pid.update(-pitch)
        y_err = wrap_to_pi(target_yaw - yaw)
        y_cmd = self.yaw_pid.update(y_err, is_angular=True)

        wx_cmd = float(np.clip(self.vx_pid.update(move_x), -1, 1))
        wy_cmd = float(np.clip(self.vy_pid.update(move_y), -1, 1))

        u_roll = float(np.clip(r_cmd + wx_cmd, -1, 1))
        u_pitch = float(np.clip(p_cmd + wy_cmd, -1, 1))
        u_yaw = float(np.clip(y_cmd, -1, 1))

        vz_out = float(self.vz_pid.update(target_vz - vz))
        throttle = clamp(0.3 + vz_out, 0.0, 1.0)

        legs_cmd = 1.0 if alt < 6.0 else -1.0

        action = np.zeros(16, dtype=np.float32)
        action[0] = throttle
        action[1] = 0.0
        action[2] = 0.0

        rcs = np.zeros(12, dtype=float)
        rcs[0] = clamp(max(0.0, u_roll), 0.0, 1.0)
        rcs[1] = clamp(max(0.0, -u_roll), 0.0, 1.0)
        rcs[2] = clamp(max(0.0, u_pitch), 0.0, 1.0)
        rcs[3] = clamp(max(0.0, -u_pitch), 0.0, 1.0)
        rcs[8] = clamp(max(0.0, u_yaw), 0.0, 1.0)
        rcs[9] = clamp(max(0.0, -u_yaw), 0.0, 1.0)
        action[3:15] = (rcs * 2.0 - 1.0).astype(np.float32)
        action[15] = float(legs_cmd)

        debug = {
            "phase": float(self._mission_phase),
            "dist_h": dist_h,
            "alt": alt,
            "target_vz": target_vz,
            "target_yaw": math.degrees(target_yaw),
            "vz": vz,
            "r_deg": math.degrees(r),
            "p_deg": math.degrees(pitch),
            "yaw_deg": math.degrees(yaw),
            "fuel": fuel,
            "stabilized": 1.0 if self._stabilized else 0.0,
            "contacts": float(np.max(contacts)),
        }
        return action, debug

