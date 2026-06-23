import gymnasium as gym
import time
from gymnasium import spaces
# pyrefly: ignore [missing-import]
import pybullet as p
import pybullet_data
import numpy as np
import os
import math
import random
from typing import Optional, Dict, Any, Tuple, List

class RocketLanderEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: Optional[str] = None, normalize_obs: bool = True, randomize_spawn: bool = True, mission_mode: bool = False):
        super(RocketLanderEnv, self).__init__()
        
        self.render_mode = render_mode
        self.normalize_obs = normalize_obs
        self.randomize_spawn = randomize_spawn
        self.mission_mode = mission_mode
        self.STEPS_PER_CONTROL = 4  # 60Hz Control (Physics is 240Hz)
        self.dt = (1.0 / 240.0) * self.STEPS_PER_CONTROL
        
        # Physics Parameters
        self.MAIN_ENGINE_POWER = 100000.0
        self.RCS_FORCE = 6000.0
        # Aerodynamics Constants
        self.DRAG_COEFF = 0.5
        self.GND_EFF_COEFF = 0.8  # Extra thrust multiplier at ground level
        self.GND_EFF_H_CLIP = 6.0 # Height in meters where ground effect starts
        self.gravity = -9.81
        
        # Mass Properties (for fast PID calibration: keep constant mass)
        self.MIN_MASS = 1000.0
        self.MAX_MASS = 1000.0
        self.FUEL_MASS = self.MAX_MASS - self.MIN_MASS

        # Normalization Scales
        self.POS_SCALE = 500.0
        self.VEL_SCALE = 50.0
        self.ANG_VEL_SCALE = 10.0
        self.ALT_SCALE = 500.0

        # Space Definitions
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(16,), dtype=np.float32)
        # Obs: [Pos_N(3), Quat(4), Vel_N(3), AngVel_N(3), FootContacts(4), Alt_N(1), Fuel(1)] = 19
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(19,), dtype=np.float32)
        
        # Internal State
        self.fuel = 1.0
        self.fuel_consumption_rate = 0.0001 
        self.rcs_fuel_rate = 0.00005
        self.max_steps = 30000 if mission_mode else 10000
        self.step_count = 0
        
        self.rocketId: Optional[int] = None
        self.terrain_id: Optional[int] = None
        self.pad_id: Optional[int] = None
        
        self.prev_dist_pad = 0.0
        self.prev_tilt = 0.0
        
        # Visualization State
        self.camera_target = np.array([0.0, 0.0, 10.0])
        self.camera_dist = 10.0
        
        # RCS Thruster Configuration
        self.rcs_config = [
            {"link": -1, "dir": [-1, 0, 0], "pos": [0.5, 0, 2.0], "type": "pr"},    # 0
            {"link": -1, "dir": [ 1, 0, 0], "pos": [-0.5, 0, 2.0], "type": "pr"},   # 1
            {"link": -1, "dir": [0, -1, 0], "pos": [0, 0.5, 2.0], "type": "pr"},    # 2
            {"link": -1, "dir": [0,  1, 0], "pos": [0, -0.5, 2.0], "type": "pr"},   # 3
            {"link": -1, "dir": [-1, 0, 0], "pos": [0.5, 0, -2.0], "type": "pr"},   # 4
            {"link": -1, "dir": [ 1, 0, 0], "pos": [-0.5, 0, -2.0], "type": "pr"},  # 5
            {"link": -1, "dir": [0, -1, 0], "pos": [0, 0.5, -2.0], "type": "pr"},   # 6
            {"link": -1, "dir": [0,  1, 0], "pos": [0, -0.5, -2.0], "type": "pr"},  # 7
            {"link": -1, "dir": [-1, 0, 0], "pos": [0, 0.5, 0], "type": "yaw"},     # 8
            {"link": -1, "dir": [ 1, 0, 0], "pos": [0, -0.5, 0], "type": "yaw"},    # 9
            {"link": -1, "dir": [ 1, 0, 0], "pos": [0, 0.5, 0], "type": "yaw"},     # 10
            {"link": -1, "dir": [-1, 0, 0], "pos": [0, -0.5, 0], "type": "yaw"},    # 11
        ]
        
        self._setup_pybullet()

    def _setup_pybullet(self):
        if self.render_mode == "human":
            self._bullet_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            self._bullet_client = p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.setRealTimeSimulation(0)
        p.setTimeStep(1.0 / 240.0)
        p.setGravity(0, 0, self.gravity)
        self._create_terrain()
        
        urdf_path = os.path.join(os.path.dirname(__file__), "assets", "rocket_lander.urdf")
        self.rocketId = p.loadURDF(urdf_path, [0, 0, 10], useFixedBase=False)
        p.changeDynamics(self.rocketId, -1, mass=self.MAX_MASS, linearDamping=0.001, angularDamping=0.001)
        
        self.joint_map = {}
        self.link_map = {}
        self.leg_joint_indices = []
        for i in range(p.getNumJoints(self.rocketId)):
            info = p.getJointInfo(self.rocketId, i)
            self.joint_map[info[1].decode("utf-8")] = i
            self.link_map[info[12].decode("utf-8")] = i
            if "joint_rocket_foot" in info[1].decode("utf-8"):
                self.leg_joint_indices.append(i)

    def _create_terrain(self):
        rows, cols = 256, 256
        self.mesh_scale = [5.0, 5.0, 1.0]
        h_scale = 5.0
        
        freq1, freq2 = random.uniform(0.01, 0.03), random.uniform(0.03, 0.06)
        offsets = np.random.uniform(0, 2*np.pi, 2)
        x = np.linspace(-rows/2 * self.mesh_scale[0], rows/2 * self.mesh_scale[0], rows)
        y = np.linspace(-cols/2 * self.mesh_scale[1], cols/2 * self.mesh_scale[1], cols)
        X, Y = np.meshgrid(x, y)
        
        val = np.sin(X * freq1 + offsets[0]) * np.cos(Y * freq1 + offsets[0]) * 3.0
        val += np.sin(X * freq2 + offsets[1]) * np.cos(Y * freq2 + offsets[1]) * 1.5
        val += np.random.uniform(-0.05, 0.05, val.shape)
        
        dist = np.sqrt(X**2 + Y**2)
        blend = np.clip((dist - 30.0) / 30.0, 0.0, 1.0)
        heights = val * blend * h_scale
        heights[dist < 30.0] = 0.0
        
        terrain_shape = p.createCollisionShape(p.GEOM_HEIGHTFIELD, meshScale=self.mesh_scale, 
                                              heightfieldTextureScaling=(rows - 1)/2,
                                              heightfieldData=heights.flatten(),
                                              numHeightfieldRows=rows, numHeightfieldColumns=cols)
        self.terrain_id = p.createMultiBody(0, terrain_shape, basePosition=[0, 0, (np.max(heights)+np.min(heights))/2.0])
        p.changeVisualShape(self.terrain_id, -1, rgbaColor=[0.7, 0.7, 0.72, 1], textureUniqueId=-1)

        pad_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=1.5, length=0.1, rgbaColor=[0.8, 0.1, 0.1, 0.8])
        self.pad_id = p.createMultiBody(baseVisualShapeIndex=pad_visual, basePosition=[0, 0, 0.05])

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        # Default starting state
        initial_pos = [0.0, 0.0, 50.0]
        initial_orn = [0.0, 0.0, 0.0, 1.0]
        initial_vel = [0.0, 0.0, 0.0]
        initial_ang_vel = [0.0, 0.0, 0.0]

        # Support for curriculum learning via options
        spawn_radius = 200.0
        if options and "spawn_radius" in options:
            spawn_radius = options["spawn_radius"]

        if self.randomize_spawn:
            # Random horizontal offset
            initial_pos[0] = random.uniform(-spawn_radius, spawn_radius)
            initial_pos[1] = random.uniform(-spawn_radius, spawn_radius)
            # Random altitude (30m to 80m)
            initial_pos[2] = random.uniform(30, 80)
            # Random initial velocity (up to 10m/s if spawn is far)
            v_scale = min(1.0, spawn_radius / 100.0)
            initial_vel = [random.uniform(-10*v_scale, 10*v_scale), 
                           random.uniform(-10*v_scale, 10*v_scale), 
                           random.uniform(-5, 5)]
            # Random initial tilt (up to 20 degrees if spawn is far)
            tilt_angle = random.uniform(0, math.radians(10 + 10*v_scale))
            tilt_axis = np.random.normal(size=3)
            tilt_axis[2] = 0
            tilt_axis /= (np.linalg.norm(tilt_axis) + 1e-6)
            initial_orn = p.getQuaternionFromAxisAngle(tilt_axis, tilt_angle)
            initial_ang_vel = [random.uniform(-0.3, 0.3), random.uniform(-0.3, 0.3), random.uniform(-0.1, 0.1)]
        
        if options:
            if "initial_pos" in options: 
                initial_pos = options["initial_pos"]
                # Disable random velocity if pos is fixed
                initial_vel = [0.0, 0.0, 0.0]
                initial_ang_vel = [0.0, 0.0, 0.0]
            if "initial_orn" in options: initial_orn = options["initial_orn"]
            if "initial_vel" in options: initial_vel = options["initial_vel"]
            if "initial_ang_vel" in options: initial_ang_vel = options["initial_ang_vel"]
            
        print(f"DEBUG: Reset Initial Pos: {initial_pos}")
        p.resetBasePositionAndOrientation(self.rocketId, initial_pos, initial_orn)
        p.resetBaseVelocity(self.rocketId, initial_vel, initial_ang_vel)
        
        self.fuel = 1.0
        self.step_count = 0
        p.changeDynamics(self.rocketId, -1, mass=self.MAX_MASS)
        
        for idx in self.leg_joint_indices:
            p.resetJointState(self.rocketId, idx, -math.pi/2)
            p.setJointMotorControl2(self.rocketId, idx, p.POSITION_CONTROL, -math.pi/2, force=10000)

        # Clear state variables for shaping
        # Use raw values to avoid dependency on POS_SCALE/VEL_SCALE in reward logic
        self.prev_dist_pad = np.linalg.norm(initial_pos[0:2])
        r, p_ang, _ = p.getEulerFromQuaternion(initial_orn)
        self.prev_tilt = np.linalg.norm([r, p_ang])
    
        # Reset camera to follow rocket immediately
        self.camera_target = np.array(initial_pos)
        self.start_real_time = time.time()
        
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        pos, quat = p.getBasePositionAndOrientation(self.rocketId)
        vel, world_ang_vel = p.getBaseVelocity(self.rocketId)
        rot_matrix = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        local_ang_vel = rot_matrix.T @ np.array(world_ang_vel)
        
        foot_contacts = []
        altitude = pos[2]
        if self.terrain_id is not None:
            for link_name in ["rocket_foot_x_plus", "rocket_foot_x_neg", "rocket_foot_y_plus", "rocket_foot_y_neg"]:
                if link_name in self.link_map:
                    c = p.getContactPoints(self.rocketId, self.terrain_id, linkIndexA=self.link_map[link_name])
                    foot_contacts.append(1.0 if len(c) > 0 else 0.0)
                else: foot_contacts.append(0.0)
        
        if self.normalize_obs:
            obs = np.concatenate([
                np.array(pos) / self.POS_SCALE,
                quat,
                np.array(vel) / self.VEL_SCALE,
                local_ang_vel / self.ANG_VEL_SCALE,
                foot_contacts,
                [altitude / self.ALT_SCALE, self.fuel]
            ]).astype(np.float32)
        else:
            obs = np.concatenate([
                pos, quat, vel, local_ang_vel, 
                foot_contacts, [altitude, self.fuel]
            ]).astype(np.float32)
        return obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # 1. Action Parse
        throttle_cmd = np.clip((action[0] + 1.0) / 2.0, 0.0, 1.0)
        gimbal_p = np.clip(action[1] * 0.35, -0.35, 0.35)
        gimbal_r = np.clip(action[2] * 0.35, -0.35, 0.35)
        legs_cmd = action[15]
        
        if self.fuel <= 0:
            throttle_cmd = 0.0
            action[3:15] = 0.0
        
        # 2. Variable Mass & Fuel
        rcs_powers = np.clip((action[3:15] + 1.0) / 2.0, 0.0, 1.0)
        # Keep fuel constant during calibration mode
        self.fuel = 1.0
        curr_m = self.MIN_MASS
        r, L = 0.5, 5.0
        i_v = 0.5 * curr_m * (r**2)
        i_l = (1.0/12.0) * curr_m * (L**2) + (1.0/4.0) * curr_m * (r**2)
        p.changeDynamics(self.rocketId, -1, mass=curr_m, localInertiaDiagonal=[i_l, i_l, i_v])
        
        # 3. Controls
        leg_tgt = -math.pi/2 + (1.2 - (-math.pi/2)) * (legs_cmd + 1.0) / 2.0
        for idx in self.leg_joint_indices:
            p.setJointMotorControl2(self.rocketId, idx, p.POSITION_CONTROL, targetPosition=leg_tgt, force=10000)
        
        p.setJointMotorControl2(self.rocketId, self.joint_map["joint_rocket_thruster_main_engine_pitch"], p.POSITION_CONTROL, gimbal_p, force=50000)
        p.setJointMotorControl2(self.rocketId, self.joint_map["joint_rocket_thruster_main_engine_roll"], p.POSITION_CONTROL, gimbal_r, force=50000)

        # 4. Dynamics Loop
        m_ratio = curr_m / self.MAX_MASS
        f_mag = throttle_cmd * self.MAIN_ENGINE_POWER
        e_id = self.link_map["rocket_thruster_main_engine"]

        for _ in range(self.STEPS_PER_CONTROL):
            if f_mag > 0:
                s = p.getLinkState(self.rocketId, e_id)
                f_w, _ = p.multiplyTransforms([0,0,0], s[1], [0,0,f_mag], [0,0,0,1])
                p.applyExternalForce(self.rocketId, -1, f_w, s[0], p.WORLD_FRAME)
            
            for i, cfg in enumerate(self.rcs_config):
                if rcs_powers[i] > 0.1:
                    mag = (2000.0 if cfg["type"] == "yaw" else self.RCS_FORCE) * m_ratio
                    f = [d * rcs_powers[i] * mag for d in cfg["dir"]]
                    p.applyExternalForce(self.rocketId, -1, f, cfg["pos"], p.LINK_FRAME)

            # Drag
            lv, _ = p.getBaseVelocity(self.rocketId)
            spd = np.linalg.norm(lv)
            pos, _ = p.getBasePositionAndOrientation(self.rocketId)
            
            if spd > 0.1:
                p.applyExternalForce(self.rocketId, -1, -self.DRAG_COEFF * spd * np.array(lv), pos, p.WORLD_FRAME)
                
            # Ground Effect (Exhaust cushion)
            if pos[2] < self.GND_EFF_H_CLIP:
                multiplier = 1.0 + self.GND_EFF_COEFF * (self.GND_EFF_H_CLIP - pos[2]) / self.GND_EFF_H_CLIP
                up_force = [0, 0, (multiplier - 1.0) * curr_m * 9.8]
                p.applyExternalForce(self.rocketId, -1, up_force, pos, p.WORLD_FRAME)
                
            p.stepSimulation()
            
        if self.render_mode == "human":
            curr_pos, curr_quat = p.getBasePositionAndOrientation(self.rocketId)
            self.camera_target = np.array(curr_pos) # Lock exactly to rocket to prevent visual lag/jitter
            target_dist = min(50.0, 10.0 + curr_pos[2] * 0.04)
            self.camera_dist = 0.9 * self.camera_dist + 0.1 * target_dist
            p.resetDebugVisualizerCamera(cameraDistance=self.camera_dist, cameraYaw=45, cameraPitch=-25, cameraTargetPosition=self.camera_target)
            time.sleep(1.0 / 60.0) # Render at 60Hz to prevent PyBullet GUI stuttering
            
        obs = self._get_obs()
        self.step_count += 1
        
        # --- Mission mode: no reward, no auto-termination ---
        if self.mission_mode:
            truncated = self.step_count >= self.max_steps
            # Only terminate on extreme OOB
            if self.normalize_obs:
                pos_check = obs[0:3] * self.POS_SCALE
            else:
                pos_check = obs[0:3]
            oob = (np.linalg.norm(pos_check[0:2]) > 2500.0 or
                   pos_check[2] > 1500.0 or pos_check[2] < -50.0)
            return obs, 0.0, oob, truncated, {}
        
        # --- Standard RL mode (unchanged) ---
        reward = self._compute_reward(obs, action)
        
        # 5. Terminations
        if self.normalize_obs:
            pos = obs[0:3] * self.POS_SCALE
            vel = obs[7:10] * self.VEL_SCALE
            spd = np.linalg.norm(vel)
            r, p_ang, _ = p.getEulerFromQuaternion(obs[3:7])
            tilt = np.linalg.norm([r, p_ang])
            has_contact = np.sum(obs[13:17]) > 0
            dist_pad = np.linalg.norm(pos[0:2])
        else:
            pos = obs[0:3]
            vel = obs[7:10]
            spd = np.linalg.norm(vel)
            r, p_ang, _ = p.getEulerFromQuaternion(obs[3:7])
            tilt = np.linalg.norm([r, p_ang])
            has_contact = np.sum(obs[13:17]) > 0
            dist_pad = np.linalg.norm(pos[0:2])
        
        terminated = False
        truncated = self.step_count >= self.max_steps
        
        if has_contact and (spd > 15.0 or tilt > math.radians(45)):
            terminated = True; reward -= 500.0
        
        landing_ok = has_contact and spd < 2.5 and tilt < math.radians(8) and dist_pad < 15.0 and pos[2] < 10.0
        if landing_ok:
            terminated = True; reward += 2000.0 + (self.fuel * 500.0)
            
        if dist_pad > 1500 or pos[2] > 1500 or pos[2] < -50:
            terminated = True; reward -= 200.0
            
        return obs, reward, terminated, truncated, {}

    def _compute_reward(self, obs: np.ndarray, action: np.ndarray) -> float:
        if self.normalize_obs:
            pos = obs[0:3] * self.POS_SCALE
            vel = obs[7:10] * self.VEL_SCALE
            dist_h = np.linalg.norm(pos[0:2])
            r, p_ang, _ = p.getEulerFromQuaternion(obs[3:7])
            tilt = np.linalg.norm([r, p_ang])
        else:
            pos = obs[0:3]
            vel = obs[7:10]
            dist_h = np.linalg.norm(pos[0:2])
            r, p_ang, _ = p.getEulerFromQuaternion(obs[3:7])
            tilt = np.linalg.norm([r, p_ang])
        
        reward = 0.0
        # 1. Shaping: Distance reduction (1.0 meter = 1.0 reward)
        reward += (self.prev_dist_pad - dist_h) * 1.0
        self.prev_dist_pad = dist_h
        
        # 2. Shaping: Tilt reduction (Aggressive scaling)
        reward += (self.prev_tilt - tilt) * 5.0
        self.prev_tilt = tilt
        
        # 3. Dynamic Guidance (Inspired by Mission V1)
        alt = pos[2]
        
        # 3.1 Horizontal Guidance (Velocity Vector Target)
        # Calculate unit vector towards pad (at 0,0)
        dist_h = np.linalg.norm(pos[0:2])
        if dist_h > 0.1:
            dir_to_pad = -pos[0:2] / dist_h
            # Target speed: faster when far, slow down when near
            target_speed_h = np.clip(0.1 * dist_h, 2.0, 15.0)
            target_vel_h = dir_to_pad * target_speed_h
        else:
            target_vel_h = np.array([0.0, 0.0])
            
        # Horizontal guidance penalty (reward for matching target velocity)
        reward -= 0.1 * np.linalg.norm(vel[0:2] - target_vel_h)
        
        # 3.2 Vertical Guidance (Glideslope)
        if alt > 50.0:
            target_vz = -5.0 # Steady approach
        else:
            # Linear glide slope to -0.5m/s at touchdown
            target_vz = -np.clip(0.15 * alt, 0.5, 5.0)
        
        # Vz penalty (Weighted heavily as it's critical for landing)
        reward -= 0.15 * abs(vel[2] - target_vz)
        
        # 3.3 Stability Guidance (Scales with proximity to ground AND distance to pad)
        # If far from pad, allow more tilt to move aggressively
        authority_mask = np.clip(1.0 - dist_h / 100.0, 0.0, 1.0) # 0 if far, 1 if near
        proximity = np.clip(1.0 - alt / 50.0, 0.0, 1.0)
        
        # Horizontal stabilization penalty (Relaxed when far or high)
        h_penalty_coeff = (0.01 + 0.09 * proximity) * authority_mask
        reward -= h_penalty_coeff * np.linalg.norm(vel[0:2])
        
        # Tilt stabilization penalty (Relaxed when far to allow maneuvering)
        tilt_penalty_coeff = (0.05 + 0.25 * proximity) * authority_mask
        reward -= tilt_penalty_coeff * tilt
        
        # 3.4 Mastery Bonus (Continuously reward precision near pad)
        if alt < 10.0 and dist_h < 5.0:
            spd_h = np.linalg.norm(vel[0:2])
            if spd_h < 1.0: reward += 0.2
            if tilt < math.radians(5): reward += 0.2
            
        reward -= 0.001 * dist_h # Small persistent attractor
        reward -= 0.001 * np.sum(np.abs(action))
        
        return float(reward)

    def render(self, mode="human"):
        if self.render_mode is None: return None
        
        curr_pos, curr_quat = p.getBasePositionAndOrientation(self.rocketId)
        self.camera_target = 0.7 * self.camera_target + 0.3 * np.array(curr_pos)
        self.camera_dist = 0.7 * self.camera_dist + 0.3 * (7.5 + curr_pos[2] * 0.05)
        
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self.camera_target, distance=self.camera_dist,
            yaw=45, pitch=-25, roll=0, upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=1.33, nearVal=0.1, farVal=2000.0)
        
        if self.render_mode == "rgb_array":
            (_, _, px, _, _) = p.getCameraImage(width=640, height=480, viewMatrix=view_matrix, projectionMatrix=proj_matrix)
            rgb_array = np.array(px, dtype=np.uint8).reshape((480, 640, 4))[:, :, :3]
            return rgb_array
        elif self.render_mode == "human":
            pass
        return None

    def close(self):
        if hasattr(self, "_bullet_client"): p.disconnect(self._bullet_client)
