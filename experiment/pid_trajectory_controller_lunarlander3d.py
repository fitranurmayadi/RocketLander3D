import math
import os
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import gymnasium as gym
import numpy as np
import pybullet as p

# Import env registration
import rocket_lander  # noqa: F401


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def wrap_to_pi(x: float) -> float:
    # Wrap to [-pi, pi]
    return (x + math.pi) % (2 * math.pi) - math.pi


def quat_to_euler(quat) -> Tuple[float, float, float]:
    # pybullet uses (x,y,z,w)
    r, pitch, yaw = p.getEulerFromQuaternion(quat)
    return r, pitch, yaw


@dataclass
class PIDGains:
    kp: float
    ki: float
    kd: float
    i_limit: float = 1e9


class PID:
    def __init__(self, gains: PIDGains):
        self.gains = gains
        self.i = 0.0
        self.prev_err = 0.0
        self.has_prev = False

    def reset(self):
        self.i = 0.0
        self.prev_err = 0.0
        self.has_prev = False

    def step(self, err: float, dt: float) -> float:
        # Integral
        self.i += err * dt
        self.i = clamp(self.i, -self.gains.i_limit, self.gains.i_limit)

        # Derivative (on error)
        if not self.has_prev:
            derr = 0.0
            self.has_prev = True
        else:
            derr = (err - self.prev_err) / max(1e-9, dt)

        self.prev_err = err
        return (
            self.gains.kp * err
            + self.gains.ki * self.i
            + self.gains.kd * derr
        )


class TrajectoryReference:
    """Phase-based reference generator (LunarLander3D-like).

    This is a *template*; you must tune the numbers after verifying signs
    and axis conventions via calibration/verify scripts.
    """

    def __init__(self):
        # Tunables
        self.vx_far = 6.0
        self.vx_near = 1.0
        self.vz_fast = -7.0
        self.vz_final = -0.5
        self.alt_switch1 = 40.0
        self.alt_switch2 = 12.0

    def get_refs(self, pos: np.ndarray, vel: np.ndarray) -> Dict[str, float]:
        # pos, vel in world frame
        x, y, z = pos
        vx, vy, vz = vel
        dist_h = math.sqrt(x * x + y * y)

        # Horizontal desired velocity: move towards pad (0,0)
        if dist_h > 1e-6:
            dir_to_pad = np.array([-x, -y]) / dist_h
        else:
            dir_to_pad = np.array([0.0, 0.0])

        # Speed schedule
        if z > self.alt_switch1:
            target_speed_h = self.vx_far
        elif z > self.alt_switch2:
            t = (z - self.alt_switch2) / max(1e-6, (self.alt_switch1 - self.alt_switch2))
            target_speed_h = (1 - t) * self.vx_near + t * self.vx_far
        else:
            target_speed_h = self.vx_near

        ref_vx = float(dir_to_pad[0] * target_speed_h)
        ref_vy = float(dir_to_pad[1] * target_speed_h)

        # Vertical schedule (glideslope-ish)
        if z > self.alt_switch1:
            ref_vz = self.vz_fast
        elif z > self.alt_switch2:
            t = (z - self.alt_switch2) / max(1e-6, (self.alt_switch1 - self.alt_switch2))
            ref_vz = (1 - t) * self.vz_final + t * self.vz_fast
        else:
            ref_vz = self.vz_final

        # Desired attitude is encoded via rate references in the controller;
        # here we output only velocity refs.
        return dict(ref_vx=ref_vx, ref_vy=ref_vy, ref_vz=float(ref_vz), dist_h=dist_h)


class PIDTrajectoryController:
    """Outer (attitude) PID -> inner (rate) PID -> action mixing.

    IMPORTANT:
    - This file is intentionally a controller *skeleton*.
    - Final performance depends on correct mapping from attitude/rate commands
      to the environment's RCS thruster directions (sign/axis).
    """

    def __init__(
        self,
        dt: float,
        attitude_gains: Dict[str, PIDGains],
        rate_gains: Dict[str, PIDGains],
        mode: str = "rcs_only",
    ):
        self.dt = dt
        self.mode = mode

        self.pid_roll_att = PID(attitude_gains["roll"])
        self.pid_pitch_att = PID(attitude_gains["pitch"])
        self.pid_yaw_att = PID(attitude_gains["yaw"])

        self.pid_wx = PID(rate_gains["wx"])
        self.pid_wy = PID(rate_gains["wy"])
        self.pid_wz = PID(rate_gains["wz"])

        self.refgen = TrajectoryReference()

        self.last_action = np.zeros(16, dtype=np.float32)

    def reset(self):
        self.pid_roll_att.reset()
        self.pid_pitch_att.reset()
        self.pid_yaw_att.reset()
        self.pid_wx.reset()
        self.pid_wy.reset()
        self.pid_wz.reset()
        self.last_action[:] = 0.0

    def attitude_to_rates(self, roll: float, pitch: float, yaw: float, refs: Dict[str, float]) -> Tuple[float, float, float]:
        # In a typical LunarLander3D-like approach:
        # outer loop tries to convert horizontal velocity error -> desired tilt angles.
        # Then tilt angle PID converts desired angles -> desired body rates.

        # Here we approximate a mapping by making target attitude:
        # - For small angles: ax ~ g * pitch, ay ~ -g * roll (convention dependent!)
        # - We instead use rate commands directly from velocity errors by a simplified
        #   heuristic: target tilt proportional to velocity error.

        # Velocity errors in XY
        err_vx = refs["ref_vx"]
        err_vy = refs["ref_vy"]
        # NOTE: This controller assumes refgen outputs velocity in world frame towards pad.
        # Replace this mapping with your calibrated mapping/sign from verification scripts.

        # Heuristic target angles (radians)
        g = 9.81
        max_tilt = math.radians(30.0)

        # Using sign convention placeholders.
        target_pitch = clamp(err_vx / max(1e-6, g), -max_tilt, max_tilt)
        target_roll = clamp(-err_vy / max(1e-6, g), -max_tilt, max_tilt)

        # Yaw target: 0 (keep yaw aligned). If you need yaw steering, extend reference.
        target_yaw = 0.0

        # Attitude errors with wrap for yaw
        err_roll = target_pitch * 0.0 + (target_roll - roll)
        err_pitch = target_pitch - pitch
        err_yaw = wrap_to_pi(target_yaw - yaw)

        # Outer PID => desired body rates
        wx_ref = self.pid_roll_att.step(err_roll, self.dt)
        wy_ref = self.pid_pitch_att.step(err_pitch, self.dt)
        wz_ref = self.pid_yaw_att.step(err_yaw, self.dt)

        return float(wx_ref), float(wy_ref), float(wz_ref)

    def rates_to_actuation(self, wx: float, wy: float, wz: float, wx_ref: float, wy_ref: float, wz_ref: float) -> Tuple[float, float, float]:
        u_roll = self.pid_wx.step(wx_ref - wx, self.dt)
        u_pitch = self.pid_wy.step(wy_ref - wy, self.dt)
        u_yaw = self.pid_wz.step(wz_ref - wz, self.dt)
        return float(u_roll), float(u_pitch), float(u_yaw)

    def mix_to_action(self, u_roll: float, u_pitch: float, u_yaw: float, legs_cmd: float = -1.0) -> np.ndarray:
        # Action vector: 16 dims.
        # rocket_lander_env.py mapping:
        # action[0]=throttle (-1..1)->0..1
        # action[1]=gimbal pitch, action[2]=gimbal roll
        # action[3:15] = RCS thrusters (11 dims here; action_space says 16 but step uses [3:15] => 12 entries)
        # action[15] = legs cmd

        # We default main engine throttle OFF for attitude stability tuning.
        # Legs cmd uses raw action[15] in [-1,1].
        action = np.zeros(16, dtype=np.float32)
        action[0] = -1.0  # throttle off

        if self.mode == "rcs_only":
            # gimbals off
            action[1] = 0.0
            action[2] = 0.0
        else:
            # allow gimbal usage (not calibrated here)
            action[1] = clamp(u_pitch, -0.35, 0.35)
            action[2] = clamp(u_roll, -0.35, 0.35)

        # RCS mixing placeholder:
        # u_roll/u_pitch/u_yaw must be mapped to the 12 RCS thrusters in env.
        # For now, we use a simple sign-based distribution.
        # Replace with the exact mapping from verify_gimbal_polarity / verify_rcs_physics.

        # action[3:15] has length 12
        rcs = np.zeros(12, dtype=np.float32)

        # Roll torque: use first 8 thrusters (pr) symmetrically
        rcs[0] = clamp(max(0.0, u_roll), 0.0, 1.0)
        rcs[1] = clamp(max(0.0, -u_roll), 0.0, 1.0)

        # Pitch torque
        rcs[2] = clamp(max(0.0, u_pitch), 0.0, 1.0)
        rcs[3] = clamp(max(0.0, -u_pitch), 0.0, 1.0)

        # Yaw torque: use yaw thrusters indices 8..11
        rcs[8] = clamp(max(0.0, u_yaw), 0.0, 1.0)
        rcs[9] = clamp(max(0.0, -u_yaw), 0.0, 1.0)

        # Convert rcs command from [0..1] power to env action space [-1..1]
        # env: rcs_powers = (action[3:15]+1)/2 => 0..1
        action[3:15] = rcs * 2.0 - 1.0

        action[15] = legs_cmd
        return action

    def act(self, obs: np.ndarray) -> np.ndarray:
        pos = obs[0:3] * 1.0
        quat = obs[3:7]
        vel = obs[7:10]
        ang_vel_local = obs[10:13]
        # foot_contacts = obs[13:17]
        alt = float(obs[17])

        r, pitch, yaw = quat_to_euler(quat)
        wx, wy, wz = ang_vel_local

        refs = self.refgen.get_refs(pos=pos, vel=vel)

        # Decide legs deploy
        legs_cmd = -1.0
        if alt < 6.0:
            legs_cmd = 1.0

        # Outer -> desired rates
        wx_ref, wy_ref, wz_ref = self.attitude_to_rates(roll=r, pitch=pitch, yaw=yaw, refs=refs)

        # Inner -> control efforts
        u_roll, u_pitch, u_yaw = self.rates_to_actuation(wx=wx, wy=wy, wz=wz,
                                                           wx_ref=wx_ref, wy_ref=wy_ref, wz_ref=wz_ref)

        action = self.mix_to_action(u_roll=u_roll, u_pitch=u_pitch, u_yaw=u_yaw, legs_cmd=legs_cmd)
        self.last_action = action
        return action


def run(difficulty: str = "easy", render: bool = False, seed: int = 0, max_steps: int = 8000):
    render_mode = "human" if render else None
    env = gym.make("RocketLander-v0", render_mode=render_mode, normalize_obs=True, randomize_spawn=False)

    # Difficulty settings for initial state (template)
    if difficulty == "easy":
        spawn_radius = 40.0
    elif difficulty == "medium":
        spawn_radius = 120.0
    else:
        spawn_radius = 250.0

    # Use env options to request initial pos/orientation if supported.
    # If you want automatic scenarios like examples, extend this.

    # Choose a deterministic initial condition around the pad
    rng = np.random.default_rng(seed)
    initial_pos = [float(rng.uniform(-spawn_radius, spawn_radius)), float(rng.uniform(-spawn_radius, spawn_radius)), 55.0]
    # Small random tilt
    tilt = float(math.radians(rng.uniform(-8, 8)))
    # Axis in XY plane
    axis = np.array([0.0, 1.0, 0.0], dtype=float)
    initial_orn = p.getQuaternionFromAxisAngle(axis, tilt)

    obs, _ = env.reset(options={"initial_pos": initial_pos, "initial_orn": initial_orn})

    # Controller dt should match env control rate
    dt = (1.0 / 240.0) * 4

    # Placeholder gains: copy from pid_stabilization_calibration.py in your follow-up tuning.
    attitude_gains = {
        "roll": PIDGains(kp=2.0, ki=0.0, kd=0.3),
        "pitch": PIDGains(kp=2.0, ki=0.0, kd=0.3),
        "yaw": PIDGains(kp=1.0, ki=0.0, kd=0.2),
    }
    rate_gains = {
        "wx": PIDGains(kp=0.8, ki=0.0, kd=0.05, i_limit=1.0),
        "wy": PIDGains(kp=0.8, ki=0.0, kd=0.05, i_limit=1.0),
        "wz": PIDGains(kp=0.7, ki=0.0, kd=0.04, i_limit=1.0),
    }

    ctrl = PIDTrajectoryController(dt=dt, attitude_gains=attitude_gains, rate_gains=rate_gains, mode="rcs_only")

    success = False
    for t in range(max_steps):
        action = ctrl.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        if t % 100 == 0:
            pos = obs[0:3]
            alt = float(obs[17])
            quat = obs[3:7]
            r, pitch, yaw = quat_to_euler(quat)
            print(f"t={t*dt:.2f}s alt={alt:.2f} roll={math.degrees(r):.1f} pitch={math.degrees(pitch):.1f} yaw={math.degrees(yaw):.1f} reward={reward:.1f}")

        if terminated or truncated:
            success = (reward > 1000) or terminated
            break

    print("Finished.")
    env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--difficulty", type=str, default="easy")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run(difficulty=args.difficulty, render=args.render, seed=args.seed)

