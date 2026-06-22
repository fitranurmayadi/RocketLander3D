import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet
import pybullet_data
import time
import os
import random
import math
from pybullet_utils import bullet_client

class LunarLander3DEnv(gym.Env):
    """
    Environment Lunar Lander 3D dengan 21 aktuator - VERSI FIXED:
      - 1 untuk Main Thruster
      - 20 untuk RCS thrusters (4 grup x 5 nozzle per grup)
      
    FIX #1: Expanded crash boundaries from 0.75 to 100.0 to match spawn range
    FIX #2: Potential-based reward shaping (no accumulated distance penalties)
      
    Observasi: 34 dimensi (hasil konkatenasi: previous 17-dim + current 17-dim)
    Aksi: continuous, vektor 21-dimensi dalam rentang [-1, 1].

    Mode planet: 'earth', 'moon', dan 'mars' yang akan menyesuaikan parameter dynamics.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    def __init__(self, render_mode=None, max_thrust=22000.0, truncation_steps=60000, drag_coeff=0.5, wind_force=15, wind_freq=0.1, planet="moon"):
        super(LunarLander3DEnv, self).__init__()
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.render_fps = self.metadata["render_fps"]
        self.max_thrust = max_thrust
        self.initial_fuel = truncation_steps * 1000  # Rescaled fuel consumption
        self.fuel = self.initial_fuel
        self.truncation_steps = truncation_steps
        self.sim_time = 1 / self.render_fps

        # Set parameter dinamik berdasarkan mode planet
        self.planet = planet.lower()
        if self.planet == "earth":
            self.gravity = -9.8
            self.drag_coeff = drag_coeff
            self.wind_force = wind_force
            self.wind_freq = wind_freq
        elif self.planet == "moon":
            self.gravity = -1.62
            self.drag_coeff = 0.0
            self.wind_force = 0.0
            self.wind_freq = 0.0
        elif self.planet == "mars":
            self.gravity = -3.711
            self.drag_coeff = 0.3
            self.wind_force = 10
            self.wind_freq = 0.1
        else:
            raise ValueError("Planet harus 'earth', 'moon', atau 'mars'")

        # Parameter observasi asli (17 dimensi) untuk normalisasi - SCALED for Apollo scale
        # Increased to 2km to prevent clipping during high-altitude starts
        pos_low, pos_high = np.array([-2000, -2000, -2000]), np.array([2000, 2000, 2000]) 
        orient_low, orient_high = -np.pi * np.ones(3), np.pi * np.ones(3)
        lin_vel_low, lin_vel_high = -200 * np.ones(3), 200 * np.ones(3) # Doubled velocity range
        ang_vel_low, ang_vel_high = -10 * np.ones(3), 10 * np.ones(3)
        fuel_low, fuel_high = np.array([0]), np.array([self.initial_fuel])
        contact_low, contact_high = np.zeros(4), 1000000 * np.ones(4)
        self.obs_min = np.concatenate([pos_low, orient_low, lin_vel_low, ang_vel_low, fuel_low, contact_low])
        self.obs_max = np.concatenate([pos_high, orient_high, lin_vel_high, ang_vel_high, fuel_high, contact_high])
        
        # Observasi penuh: konkatenasi dari observasi sebelumnya dan saat ini (34 dimensi)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(34,), dtype=np.float32)
        
        # Hanya mode aksi continuous (21-dimensi)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(21,), dtype=np.float32)
        
        # Setup Bullet client
        self._bullet_client = None
        self._physics_client_id = -1
        self.lander_id = None
        self.hud_ids = []

        # --- Definisi aktuator (Scaled for realistic size) ---
        # Main thruster nozzle exit position (Apollo scale: ~-6m below body center)
        self.main_thruster_local_position = np.array([0, 0, -6.0])  # Nozzle exit at scaled position
        self.main_thruster_local_direction = np.array([0, 0, 1])      # mendorong ke atas


        # RCS groups: 4 grup, masing-masing 5 thruster (Intuitive radial-tangential-axial pattern)
        # 0: Push Away, 1: Push CW, 2: Push CCW, 3: Push UP, 4: Push DOWN
        self.rcs_thruster_force_scale = 5000.0 # Standard for Apollo-scale 5T lander
        self.rcs_groups = {
            'front': { # Cluster at +X
                'origin': np.array([2.8, 0, 2.4]),
                'offsets': [np.array([0.6, 0, 0])] * 5, 
                'force_directions': [
                    np.array([1, 0, 0]),  # 0: Push +X (Away)
                    np.array([0, -1, 0]), # 1: Push -Y (CW)
                    np.array([0, 1, 0]),  # 2: Push +Y (CCW)
                    np.array([0, 0, 1]),  # 3: Push +Z (Up)
                    np.array([0, 0, -1])  # 4: Push -Z (Down)
                ]
            },
            'back': { # Cluster at -X
                'origin': np.array([-2.8, 0, 2.4]),
                'offsets': [np.array([-0.6, 0, 0])] * 5,
                'force_directions': [
                    np.array([-1, 0, 0]), # 0: Push -X (Away)
                    np.array([0, 1, 0]),  # 1: Push +Y (CW)
                    np.array([0, -1, 0]), # 2: Push -Y (CCW)
                    np.array([0, 0, 1]),  # 3: Push +Z (Up)
                    np.array([0, 0, -1])  # 4: Push -Z (Down)
                ]
            },
            'left': { # Cluster at +Y
                'origin': np.array([0, 2.8, 2.4]),
                'offsets': [np.array([0, 0.6, 0])] * 5,
                'force_directions': [
                    np.array([0, 1, 0]),  # 0: Push +Y (Away)
                    np.array([1, 0, 0]),  # 1: Push +X (CW)
                    np.array([-1, 0, 0]), # 2: Push -X (CCW)
                    np.array([0, 0, 1]),  # 3: Push +Z (Up)
                    np.array([0, 0, -1])  # 4: Push -Z (Down)
                ]
            },
            'right': { # Cluster at -Y
                'origin': np.array([0, -2.8, 2.4]),
                'offsets': [np.array([0, -0.6, 0])] * 5,
                'force_directions': [
                    np.array([0, -1, 0]), # 0: Push -Y (Away)
                    np.array([-1, 0, 0]), # 1: Push -X (CW)
                    np.array([1, 0, 0]),  # 2: Push +X (CCW)
                    np.array([0, 0, 1]),  # 3: Push +Z (Up)
                    np.array([0, 0, -1])  # 4: Push -Z (Down)
                ]
            }
        }

        # Contoh tambahan: beberapa thruster untuk debugging (jika diperlukan)
        self.thruster_local_positions = [
            np.array([0.55, 0.55, 0.6]),
            np.array([-0.55, 0.55, 0.6]),
            np.array([0.55, -0.55, 0.6]),
            np.array([-0.55, -0.55, 0.6])
        ]
        self.current_thruster_forces = np.zeros(len(self.thruster_local_positions))
        
        # FIX #2: Track previous distance for potential-based shaping
        self.prev_distance_to_target = None
        
        # Camera tracking settings
        self.camera_distance = 10.0
        self.camera_pitch = -35
        self.camera_yaw = 45
        
        # Curriculum Learning: Dynamic Spawn Radius
        self.spawn_radius = 500.0  # Default to mid-range
        
        self._load_lander()
        self.prev_obs = np.zeros(17, dtype=np.float32)  # inisialisasi observasi sebelumnya

    def set_spawn_radius(self, radius):
        """Set the horizontal spawn radius for curriculum learning."""
        self.spawn_radius = max(5.0, min(800.0, float(radius))) # Expanded to match 1km terrain
        print(f"Environment spawn radius set to: {self.spawn_radius}m")

    def _load_lander(self, initial_pos=None, initial_orient=None):
        if self._bullet_client is None:
            if self.render_mode == "human":
                self._bullet_client = bullet_client.BulletClient(pybullet.GUI, options="--width=1080 --height=1080")
                self._bullet_client.configureDebugVisualizer(self._bullet_client.COV_ENABLE_GUI, 0)
            else:
                self._bullet_client = bullet_client.BulletClient(pybullet.DIRECT)
        self._init_physics_client(initial_pos=initial_pos, initial_orient=initial_orient)

    def _create_heightfield(self):
        """Generates a procedural heightfield for Moon/Mars terrain - SCALED for Apollo."""
        self.heightfield_rows = 256
        self.heightfield_cols = 256
        self.height_scale = 5.0    # Gentler hills
        self.mesh_scale = [5.0, 5.0, 1.0] # 5m spacing = 1.28km x 1.28km area
        
        # Generate random height data smoothly
        heights = np.zeros((self.heightfield_rows, self.heightfield_cols))
        
        # Create hills using low-frequency sine waves for natural look.
        freq1 = random.uniform(0.01, 0.03) # Much lower frequency for gentle hills
        freq2 = random.uniform(0.03, 0.06)
        offsets = np.random.uniform(0, 2*np.pi, 2)
        
        # Vectorized Generation using NumPy
        # Create grid coordinates
        X_vals = np.linspace(-self.heightfield_rows/2 * self.mesh_scale[0], self.heightfield_rows/2 * self.mesh_scale[0], self.heightfield_rows)
        Y_vals = np.linspace(-self.heightfield_cols/2 * self.mesh_scale[1], self.heightfield_cols/2 * self.mesh_scale[1], self.heightfield_cols)
        X, Y = np.meshgrid(X_vals, Y_vals)
        
        # Calculate distance from center (flat pad zone)
        dist = np.sqrt(X**2 + Y**2)
        
        # Generate Noise
        val = np.sin(X * freq1 + offsets[0]) * np.cos(Y * freq1 + offsets[0]) * 3.0
        val += np.sin(X * freq2 + offsets[1]) * np.cos(Y * freq2 + offsets[1]) * 1.5
        # Small micro-roughness
        val += np.random.uniform(-0.05, 0.05, val.shape)
        
        # Apply Blend/Flatten logic
        # Blend factor: 0.0 at r<=30, ramping to 1.0 at r>=60
        # Larger pad needed for 11m wide lander!
        blend = np.clip((dist - 30.0) / 30.0, 0.0, 1.0)
        
        # Final height
        heights = val * blend * self.height_scale
        
        # Ensure exact flat center
        heights[dist < 30.0] = 0.0

        # Flatten for PyBullet
        heights_flattened = heights.flatten()
        
        # Create Physics Body
        terrain_shape = self._bullet_client.createCollisionShape(
            shapeType=pybullet.GEOM_HEIGHTFIELD,
            meshScale=self.mesh_scale,
            heightfieldTextureScaling=(self.heightfield_rows - 1) / 2,
            heightfieldData=heights_flattened,
            numHeightfieldRows=self.heightfield_rows,
            numHeightfieldColumns=self.heightfield_cols
        )
        
        # Determine center height offset
        # PyBullet heightfields are centered vertically on their Z range.
        # We want Z=0 to be the "ground" of the landing pad.
        # Since our logic outputs height 'val' relative to zero, but PyBullet centers the AABB...
        # Actually, for GEOM_HEIGHTFIELD, the mesh is centered.
        # So we might need to adjust basePosition z.
        # However, usually placing at z=0 works if the logic generates positive/negative around 0.
        # Our pad is at 0. Let's see. If the mesh range is [-3, 3], PyBullet centers it.
        # To align Z=0 logic with Z=0 world, we typically put the body at Z=0.
        # IMPORTANT: PyBullet GEOM_HEIGHTFIELD origin is in the CENTER of the grid XY, and CENTER of Z range.
        # So if our data ranges [0, 5], center is 2.5. World Z=0 will be 2.5 in local.
        # To make "0" in data be "0" in world, we need to shift.
        
        min_h = np.min(heights)
        max_h = np.max(heights)
        mid_h = (max_h + min_h) / 2.0
        
        # We want data_Z = 0 to be world_Z = 0
        # Body Pos Z needs to compensate for the centering.
        # If data ranges [min, max], body center is at (min+max)/2 relative to data origin.
        # So placing body at Z = mid_h puts "0" at mid_h - mid_h = 0?
        # Actually usually easier to just place at 0 and let gravity settle, but for landing pad we need precision.
        # If data is 0.0 at center, and we want that to be World Z=0.
        # The mesh's local origin is at (0,0, mid_h_of_data).
        # So if we put body at Z = mid_h, the physical surface corresponding to data=0 *should* be at ???
        # Trial and error: Usually putting heightfield at Z=0 results in surface at Z ~ 0 if data is symmetric.
        # If data is 0 to 10, center is 5. Placing body at Z=0 puts the "5" level at 0. So "0" level is at -5.
        # To put "0" level at 0, we need body at Z = 5.
        
        # So body_z = mid_h is my best guess to align data=0 with world=0.
        
        self.terrain_id = self._bullet_client.createMultiBody(0, terrain_shape, basePosition=[0, 0, mid_h])
        
        # Coloring
        self._bullet_client.changeVisualShape(self.terrain_id, -1, rgbaColor=[0.7, 0.7, 0.72, 1.0], specularColor=[0.1,0.1,0.1], textureUniqueId=-1)
        
        return heights # Return data for rock placement

    def _init_physics_client(self, initial_pos=None, initial_orient=None):
        self._bullet_client.resetSimulation()
        self._bullet_client.setGravity(0, 0, self.gravity)
        self._bullet_client.setTimeStep(1.0 / self.render_fps)
        self._bullet_client.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # === TERRAIN GENERATION ===
        height_data = self._create_heightfield()
        # Remove old plane loading
        # self.plane_id = self._bullet_client.loadURDF("plane.urdf")
        self.plane_id = self.terrain_id # Alias for leg sensors
        
        # === VISUAL ENHANCEMENT ===
        if self.render_mode == "human":
            self._create_environment_visuals(height_data) # Pass height data to place rocks
            self._init_thruster_visuals()
        # ==========================
        
        # Use dynamic spawn radius if not fixed
        if initial_pos is None:
            r = self.spawn_radius
            # User wants around 500-1000m altitude
            start_pos = [random.uniform(-r, r), random.uniform(-r, r), random.uniform(500, 1000)]
        else:
            start_pos = initial_pos
            
        if initial_orient is None:
            start_orient = pybullet.getQuaternionFromEuler([
                np.radians(random.uniform(-20, 20)),
                np.radians(random.uniform(-20, 20)),
                np.radians(random.uniform(-20, 20))
            ])
        else:
            # Assume initial_orient is already a quaternion or euler
            if len(initial_orient) == 3: # Euler
                start_orient = pybullet.getQuaternionFromEuler(initial_orient)
            else:
                start_orient = initial_orient
        urdf_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "assets", "lunar_lander.urdf")
        self.lander_id = self._bullet_client.loadURDF(urdf_path,
                                                      basePosition=start_pos,
                                                      baseOrientation=start_orient)
        
        # Realistic vacuum physics: NO DAMPING
        self._bullet_client.changeDynamics(self.lander_id, -1, 
                                           linearDamping=0.0, 
                                           angularDamping=0.0,
                                           jointDamping=0.0)
        for i in range(self._bullet_client.getNumJoints(self.lander_id)):
            self._bullet_client.changeDynamics(self.lander_id, i, 
                                               linearDamping=0.0, 
                                               angularDamping=0.0,
                                               jointDamping=0.0)

        # Inisialisasi sensor kaki sesuai nama pada URDF
        self.leg_sensor_names = ["leg_front_left_sensor", "leg_back_right_sensor", 
                                 "leg_front_right_sensor", "leg_back_left_sensor"]
        self.leg_sensor_indices = {}
        for i in range(self._bullet_client.getNumJoints(self.lander_id)):
            joint_info = self._bullet_client.getJointInfo(self.lander_id, i)
            link_name = joint_info[12].decode("utf-8")
            if link_name in self.leg_sensor_names:
                self.leg_sensor_indices[link_name] = i
        self.prev_shaping = None
    
    def _create_environment_visuals(self, height_data=None):
        """Create visual elements: Landing Pad and Terrain Rocks."""
        # 1. Landing Pad (Red Circle, 5m diameter -> 2.5m radius)
        # Place slightly above 0.0 to ensure visibility over flat ground
        pad_visual = self._bullet_client.createVisualShape(
            shapeType=pybullet.GEOM_CYLINDER,
            radius=2.5,
            length=0.1,
            rgbaColor=[0.8, 0.1, 0.1, 0.8], # Red semi-transparent
            specularColor=[0.4, 0.4, 0.4]
        )
        self.pad_id = self._bullet_client.createMultiBody(
            baseVisualShapeIndex=pad_visual,
            basePosition=[0, 0, 0.05] 
        )

        # Rocks removed for performance and cleaner look
        self.rock_ids = []

    def _init_thruster_visuals(self):
        """Create visual placeholders for thrusters."""
        self.main_flame_id = -1
        self.rcs_visual_ids = []
        
        # Main Flame Visual (SCALED 4x to match lander)
        visual_shape_id = self._bullet_client.createVisualShape(
            shapeType=pybullet.GEOM_CYLINDER,
            radius=1.0,   # 4x larger: was 0.2
            length=6.0,   # 4x larger: was 1.5
            rgbaColor=[1, 0.7, 0.1, 0.0], # Orange-yellow, invisible initially
            visualFramePosition=[0, 0, -3.0], # Offset center (4x: was -0.75)
            visualFrameOrientation=[0, 0, 0, 1]
        )
        self.main_flame_id = self._bullet_client.createMultiBody(
            baseVisualShapeIndex=visual_shape_id,
            basePosition=[0, 0, -100] # Hide initially
        )
        # Disable collision for visuals
        self._bullet_client.setCollisionFilterGroupMask(self.main_flame_id, -1, 0, 0)

        # RCS Visuals disabled for performance
        self.rcs_visual_ids = []

    def _update_thruster_visuals(self, action):
        """Update position and visibility of thruster visuals."""
        if self.lander_id is None:
            return

        base_pos, base_orient = self._bullet_client.getBasePositionAndOrientation(self.lander_id)
        # Fix: getMatrixFromQuaternion returns a tuple of 9, needs reshape
        rot_matrix = np.array(self._bullet_client.getMatrixFromQuaternion(base_orient)).reshape(3, 3)

        # 1. Main Thruster
        thrust = action[0]
        if thrust > 0.01:
            # Flame position: below the nozzle, extending based on thrust
            # For Apollo scale: nozzle at -6m, flame extends 2-8m below that
            flame_offset = np.array([0, 0, -2.0 - 6.0 * thrust])  # 2-8m below nozzle
            flame_local_pos = self.main_thruster_local_position + flame_offset
            flame_world_pos = np.array(base_pos) + rot_matrix.dot(flame_local_pos)
            
            # Simple rotation match
            self._bullet_client.resetBasePositionAndOrientation(self.main_flame_id, flame_world_pos, base_orient)
            
            # Full opacity flame (orange-yellow)
            brightness = 0.5 + 0.5 * thrust
            self._bullet_client.changeVisualShape(self.main_flame_id, -1, rgbaColor=[1.0, brightness, 0.0, 1.0])
        else:
            # Hide flame
            self._bullet_client.changeVisualShape(self.main_flame_id, -1, rgbaColor=[0,0,0,0])

        # 2. RCS Thrusters - Disabled for performance

    def _apply_dynamics(self):
        # Terapkan drag dan gangguan angin
        lin_vel, ang_vel = self._bullet_client.getBaseVelocity(self.lander_id)
        drag_force = -self.drag_coeff * np.array(lin_vel)
        drag_torque = -self.drag_coeff * 0.5 * np.array(ang_vel)
        self._bullet_client.applyExternalForce(self.lander_id, -1, drag_force, [0, 0, 0], self._bullet_client.LINK_FRAME)
        self._bullet_client.applyExternalTorque(self.lander_id, -1, drag_torque, self._bullet_client.LINK_FRAME)
        noise = np.random.uniform(-20, 20, 3)
        wind_force = self.wind_force * np.sin(2 * np.pi * self.wind_freq * self.sim_time) + noise
        self._bullet_client.applyExternalForce(self.lander_id, -1, wind_force, [0, 0, 0], self._bullet_client.LINK_FRAME)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Support both mission script conventions (initial_ vs spawn_)
        initial_pos = None
        initial_orient = None
        if options:
            initial_pos = options.get("initial_pos", options.get("spawn_pos", None))
            initial_orient = options.get("initial_orient", options.get("spawn_att", None))
        
        self._load_lander(initial_pos=initial_pos, initial_orient=initial_orient)
        self.step_counter = 0
        self.fuel = self.initial_fuel
        action = np.zeros(21)  # aksi awal nol
        current_obs = self._get_obs(action)
        # Inisialisasi prev_obs dengan current_obs sehingga full observation awal konsisten
        self.prev_obs = current_obs.copy()
        
        # FIX #2: Initialize distance tracking for potential-based shaping
        pos, _, _, _, _, _ = self._unpack_obs(current_obs)
        target_pos = np.array([0.0, 0.0, 0.0])
        self.prev_distance_to_target = np.linalg.norm(pos - target_pos)
        
        full_obs = np.concatenate([current_obs, self.prev_obs])
        
        # Reset HUD item tracking
        for item_id in self.hud_ids:
            try:
                self._bullet_client.removeUserDebugItem(item_id)
            except:
                pass
        self.hud_ids = []
            
        return full_obs, self._get_info()

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._apply_thruster_forces(action)
        self._apply_dynamics()
        self._bullet_client.stepSimulation()
        # Update leg contacts from current state
        _ = self._get_contact_forces()
        
        if self.render_mode == "human":
            time.sleep(0.0000001)  # Match physics timestep for real-time visualization
        
        self.step_counter += 1

        action_main = action[0] * 0.02 # Lower fuel consumption rate
        action_rcs = np.sum(np.abs(action[1:])) * 0.005
        self.fuel = max(0, self.fuel - (action_main + action_rcs))
        
        current_obs = self._get_obs(action)
        full_obs = np.concatenate([current_obs, self.prev_obs])
        self.prev_obs = current_obs.copy()
        reward, terminated, truncated = self._get_reward(full_obs, action)
        info = self._get_info()
        
        # Safe telemetry for debugging
        if self.lander_id is not None and self._bullet_client is not None:
            pos_real, _ = self._bullet_client.getBasePositionAndOrientation(self.lander_id)
            info['real_pos_z'] = pos_real[2]

        # Update camera if in human mode to follow the lander
        if self.render_mode == "human":
            # self._update_thruster_visuals(action)  # Disabled for performance
            self.render()

        return full_obs, float(reward), terminated, truncated, info

    def _apply_thruster_forces(self, action):
        base_pos, base_orient = self._bullet_client.getBasePositionAndOrientation(self.lander_id)
        rot_matrix = np.array(self._bullet_client.getMatrixFromQuaternion(base_orient)).reshape(3, 3)
        
        # Main thruster (aksi index 0)
        main_force_mag = action[0] * self.max_thrust
        main_thruster_world_pos = np.array(base_pos) + rot_matrix.dot(self.main_thruster_local_position)
        main_thruster_force = (rot_matrix.dot(self.main_thruster_local_direction) * main_force_mag).tolist()
        self._bullet_client.applyExternalForce(self.lander_id, -1, main_thruster_force, main_thruster_world_pos.tolist(), pybullet.WORLD_FRAME)
        
        # RCS thrusters: aksi indices 1..20, per grup (front, back, left, right)
        action_idx = 1
        for group_key in ['front', 'back', 'left', 'right']:
            group = self.rcs_groups[group_key]
            group_origin_world = np.array(base_pos) + rot_matrix.dot(group['origin'])
            for i in range(5):
                nozzle_world_pos = group_origin_world + rot_matrix.dot(group['offsets'][i])
                force_mag = action[action_idx] * self.rcs_thruster_force_scale
                force_dir = rot_matrix.dot(group['force_directions'][i])
                force_vector = (force_dir * force_mag).tolist()
                self._bullet_client.applyExternalForce(self.lander_id, -1, force_vector, nozzle_world_pos.tolist(), pybullet.WORLD_FRAME)
                action_idx += 1

    def _get_contact_forces(self):
        contact_forces = np.zeros(len(self.leg_sensor_names))
        self.leg_contacts = [False] * len(self.leg_sensor_names)
        for idx, sensor_name in enumerate(self.leg_sensor_names):
            if sensor_name in self.leg_sensor_indices:
                link_index = self.leg_sensor_indices[sensor_name]
                contacts = self._bullet_client.getContactPoints(self.lander_id, self.plane_id, linkIndexA=link_index)
                total_force = sum(cp[9] for cp in contacts)
                contact_forces[idx] = total_force
                if total_force > 0.1: # Small threshold to avoid noise
                    self.leg_contacts[idx] = True
        return contact_forces

    def _get_obs(self, action):
        pos, orient = self._bullet_client.getBasePositionAndOrientation(self.lander_id)
        lin_vel, ang_vel = self._bullet_client.getBaseVelocity(self.lander_id)
        euler_orient = pybullet.getEulerFromQuaternion(orient)
        contact_forces = self._get_contact_forces()
        
        fuel_norm = self.fuel
        raw_obs = np.array(list(pos) + list(euler_orient) + list(lin_vel) + list(ang_vel) + [fuel_norm] + list(contact_forces), dtype=np.float32)
        normalized_obs = 2 * (raw_obs - self.obs_min) / (self.obs_max - self.obs_min) - 1
        return np.clip(normalized_obs, -1, 1).astype(np.float32)

    def _unpack_obs(self, obs):
        # Jika obs merupakan full observation (34-dimensi), ambil bagian current (17-dimensi)
        if obs.shape[0] == 34:
            obs = obs[17:]
        pos = obs[0:3]
        orient = obs[3:6]
        lin_vel = obs[6:9]
        ang_vel = obs[9:12]
        fuel = obs[12]
        contact = obs[13:17]
        return pos, orient, lin_vel, ang_vel, fuel, contact

    def _get_reward(self, obs, action):
        """
        FIX #2: Potential-based reward shaping
        - Reward for moving closer to target (positive)
        - No penalty for just being far away (avoiding suicide incentive)
        - Small fuel consumption penalty
        - Large bonuses/penalties for landing/crashing
        """
        # Pisahkan full observation menjadi current (17-dim) dan previous (17-dim)
        current_obs = obs[:17]
        prev_obs = obs[17:]
        
        # Ekstrak komponen-komponen state
        pos, orient, lin_vel, ang_vel, fuel, contact = self._unpack_obs(current_obs)
        
        # Target: landing pad at origin
        target_pos = np.array([0.0, 0.0, 0.0])
        
        # -------------------------------------------
        # POTENTIAL-BASED SHAPING (FIX #2)
        # -------------------------------------------
        # Calculate current distance to target
        current_distance = np.linalg.norm(pos - target_pos)
        
        # FIX #2: Potential-based shaping: reward improvement, not absolute error
        # Increased weight from 100.0 to 500.0 to make movement toward target more rewarding
        if self.prev_distance_to_target is not None:
            distance_shaping = (self.prev_distance_to_target - current_distance) * 500.0
        else:
            distance_shaping = 0.0
        
        # Update previous distance for next step
        self.prev_distance_to_target = current_distance
        
        # Additional shaping for orientation and velocity
        orientation_penalty = -0.5 * np.sum(np.abs(orient))  # Prefer upright
        velocity_penalty = -0.1 * np.linalg.norm(lin_vel)  # Prefer slow descent
        angular_velocity_penalty = -0.1 * np.linalg.norm(ang_vel)  # Prefer stable
        
        # Fuel consumption penalty (small)
        thruster_penalty = 0.01 * (abs(action[0]) + np.sum(np.abs(action)))
        
        # Total reward
        reward = distance_shaping + orientation_penalty + velocity_penalty + angular_velocity_penalty - thruster_penalty
        
        # -------------------------------------------
        # TERMINATION CONDITIONS (FIX #1: Expanded boundaries)
        # -------------------------------------------
        terminated = False
        truncated = False
        
        # Denormalize current state for clean thresholds
        raw_current = (current_obs + 1) / 2 * (self.obs_max - self.obs_min) + self.obs_min
        pos_real = raw_current[0:3]
        vel_real = raw_current[6:9]
        orient_real = raw_current[3:6]
        
        # Crash conditions
        crash_condition = (
            abs(pos_real[0]) > 1500.0 or  # Near 1.5km wall
            abs(pos_real[1]) > 1500.0 or
            pos_real[2] < -5.0 or        # Increased ground buffer for landing legs
            abs(orient_real[0]) > np.pi/2 or # Tilted 90 deg
            abs(orient_real[1]) > np.pi/2 or
            self.fuel <= 0                # Out of fuel
        )
        
        if crash_condition:
            terminated = True
            reward = -2000.0 if self.fuel > 0 else -500.0 # Severe crash vs out of fuel
            return reward, terminated, truncated
        
        # Success: Landing on pad
        velocity_magnitude_real = np.linalg.norm(vel_real)
        num_legs_touching = np.sum(self.leg_contacts)
        is_upright = abs(orient_real[0]) < 0.15 and abs(orient_real[1]) < 0.15
        
        # Precision Bonus
        dist_horiz_real = np.linalg.norm(pos_real[0:2])
        precision_bonus = 500.0 * max(0.0, 1.0 - dist_horiz_real / 50.0) # Bonus within 50m of center
        
        # Landing logic (Adjusted for scaled height: 6.64m sensor + buffer)
        if num_legs_touching >= 3 and velocity_magnitude_real < 1.5 and is_upright and pos_real[2] < 12.0:
            terminated = True
            reward = 1000.0 + precision_bonus
            return reward, terminated, truncated
        
        # Stable hover logic for "Mission Complete"
        if not hasattr(self, 'stable_hover_counter'):
            self.stable_hover_counter = 0
            
        if pos_real[2] < 12.0 and velocity_magnitude_real < 0.2 and is_upright and num_legs_touching >= 2:
            self.stable_hover_counter += 1
            if self.stable_hover_counter >= 50:
                terminated = True
                reward = 1000.0 + precision_bonus
                return reward, terminated, truncated
        else:
            self.stable_hover_counter = 0

        # Truncation: episode too long
        if self.step_counter >= self.truncation_steps:
            truncated = True
        
        return reward, terminated, truncated

    def _get_info(self):
        # Get real position and velocity for debugging/telemetry
        if self.lander_id is not None:
             pos, _ = self._bullet_client.getBasePositionAndOrientation(self.lander_id)
             lin_vel, _ = self._bullet_client.getBaseVelocity(self.lander_id)
        else:
             pos = [0, 0, 0]
             lin_vel = [0, 0, 0]
             
        return {
            "step_count": self.step_counter,
            "real_pos_z": pos[2],
            "fuel": self.fuel,
            "real_pos": np.array(pos),
            "real_vel": np.array(lin_vel)
        }
    
    def render(self):
        if self.render_mode == "human":
            lander_pos, _ = self._bullet_client.getBasePositionAndOrientation(self.lander_id)
            self._bullet_client.resetDebugVisualizerCamera(cameraDistance=20,
                                                             cameraYaw=45,
                                                             cameraPitch=-45,
                                                             cameraTargetPosition=lander_pos)
            return None
        elif self.render_mode == "rgb_array":
            lander_pos, _ = self._bullet_client.getBasePositionAndOrientation(self.lander_id)
            camera_eye = [lander_pos[0] + 15, lander_pos[1] + 15, lander_pos[2] + 10]
            camera_target = lander_pos
            camera_up = [0, 0, 1]
            view_matrix = self._bullet_client.computeViewMatrix(camera_eye, camera_target, camera_up)
            aspect = 1920 / 1080
            projection_matrix = self._bullet_client.computeProjectionMatrixFOV(fov=60, aspect=aspect, nearVal=0.1, farVal=100)
            width = 1920
            height = 1080
            img_arr = self._bullet_client.getCameraImage(width, height, view_matrix, projection_matrix)
            rgb_array = np.reshape(img_arr[2], (height, width, 4))
            rgb_array = rgb_array[:, :, :3]
            return rgb_array

    def close(self):
        if self._bullet_client is not None:
            try:
                self._bullet_client.disconnect()
            except Exception:
                pass
            self._physics_client_id = -1

    def update_hud(self, data):
        """
        Update real-time HUD using PyBullet debug text.
        'data' is a list of strings to display.
        """
        if self._bullet_client is None or self.render_mode is None:
            return
            
        # Clean old HUD items
        for item_id in self.hud_ids:
            try:
                self._bullet_client.removeUserDebugItem(item_id)
            except:
                pass
        self.hud_ids = []
        
        # Draw new HUD (Attached above the lander)
        if self.lander_id is not None:
            try:
                pos, _ = self._bullet_client.getBasePositionAndOrientation(self.lander_id)
                # We'll use a fixed vertical offset in world space
                for i, line in enumerate(data):
                    node_id = self._bullet_client.addUserDebugText(
                        line, 
                        [pos[0], pos[1], pos[2] + 8.0 - i*1.2], 
                        textColorRGB=[1, 1, 0], 
                        textSize=1.5
                    )
                    self.hud_ids.append(node_id)
            except Exception:
                pass
