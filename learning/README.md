# RocketLander3D: Learning Roadmap & Tutorial Guide

Welcome to the **RocketLander3D** educational roadmap! This folder is designed for beginners who want to learn how to control a 3D rocket simulation from scratch using **Gymnasium**, **PyBullet**, and classic **Control Theory (PID)**.

---

## 🚀 The Learning Roadmap

The tutorials are organized sequentially. Each script is fully self-contained and focuses on a single core concept:

### 1. [learning_1_gymnasium_pybullet_basic.py](file:///home/aiot/Projects/RocketLander/learning/learning_1_gymnasium_pybullet_basic.py)
* **Objective**: Understand Gymnasium environments, PyBullet physics client, observation space, action space, and basic keyboard control.
* **Theory**: How physical simulation step-loops work, rendering, and basic actuator mappings.

### 2. [learning_2_altitude_pid.py](file:///home/aiot/Projects/RocketLander/learning/learning_2_altitude_pid.py)
* **Objective**: Write a Proportional-Integral-Derivative (PID) controller to keep the rocket hovering at a target altitude.
* **Theory**:
  * PID math: $u(t) = K_p e(t) + K_i \int e(t)dt + K_d \frac{de(t)}{dt}$
  * Feedforward ($ff$): Why adding gravity compensation ($T_{\text{hover}} \approx 9.81\%$) makes the controller extremely stable.

### 3. [learning_3_horizontal_pd.py](file:///home/aiot/Projects/RocketLander/learning/learning_3_horizontal_pd.py)
* **Objective**: Command the rocket to move horizontally ($X, Y$) to a target coordinate.
* **Theory**:
  * Outer-Loop vs Inner-Loop: The horizontal controller (outer loop) calculates the required tilt angle, which the attitude controller (inner loop) executes.
  * Accelerating by Tilting: How tilting the body Z-axis projects a component of the engine thrust onto the horizontal plane.

### 4. [learning_4_euler_attitude_rcs.py](file:///home/aiot/Projects/RocketLander/learning/learning_4_euler_attitude_rcs.py)
* **Objective**: Control the rocket's roll, pitch, and yaw using Reaction Control System (RCS) gas thrusters.
* **Theory**:
  * Quaternions vs Euler Angles.
  * Euler Rate Singularity: Why yaw control becomes unstable at a $90^\circ$ pitch tilt, and how scaling the yaw error by $\cos(\text{pitch})$ stabilizes the loop.

### 5. [learning_5_xyz_euler_inverse_kinematics.py](file:///home/aiot/Projects/RocketLander/learning/learning_5_xyz_euler_inverse_kinematics.py)
* **Objective**: Understand cross-coupling and thrust projection asymmetry in XYZ Euler rotation sequence.
* **Theory**:
  * Thrust vector projection equations:
    $$\text{Thrust}_x = T \sin(\text{pitch}) \cos(\text{roll})$$
    $$\text{Thrust}_y = -T \sin(\text{roll})$$
  * Inverse Kinematics (IK): How to calculate the exact pitch and roll angles to project a perfectly symmetric horizontal acceleration vector without cross-coupling drift.

### 6. [learning_6_trajectory_planning_splines.py](file:///home/aiot/Projects/RocketLander/learning/learning_6_trajectory_planning_splines.py)
* **Objective**: Plan smooth trajectories using 3D cubic splines to guide the rocket gently.
* **Theory**:
  * Boundary conditions (position, velocity, acceleration).
  * Monotonic vertical velocity planning for a smooth descent without engine shut-downs.

### 7. [learning_7_full_rocketlander_mission.py](file:///home/aiot/Projects/RocketLander/learning/learning_7_full_rocketlander_mission.py)
* **Objective**: Combine all previous lessons into a single, complete mission state machine.
* **Flow**: Prelaunch $\rightarrow$ Ascent $\rightarrow$ Waypoint $\rightarrow$ Boostback $\rightarrow$ Landing Burn $\rightarrow$ Touchdown.

---

## 📚 Essential Theory for Beginners

### What is Gymnasium?
Gymnasium is a standard API for reinforcement learning and robotic control simulations. It provides:
1. `env.reset()`: Spawns the agent (rocket) and returns the initial state (`observation`).
2. `env.step(action)`: Applies the inputs, runs the physics for one time step ($dt$), and returns:
   * `observation`: Positions, velocities, angles, contact sensors.
   * `reward`: Reward value (useful for RL, ignored in PID control).
   * `terminated`: True if crashed or landed.
   * `truncated`: True if time limit reached.
   * `info`: Extra debug metadata.

### What is PyBullet?
PyBullet is a fast, easy-to-use Python module for physics simulation. It handles rigid body dynamics, collision detection, and gravity. In this project, the rocket is simulated as a rigid body with thruster forces applied to its center of mass and nozzle.

### PID Tuning Tip
When tuning PID gains:
1. Increase $K_p$ until the system starts oscillating, then back off slightly.
2. Increase $K_d$ to dampen the oscillations and reduce overshoot.
3. Use a small $K_i$ to remove any steady-state error (e.g. constant wind drift), but add anti-windup clamping to prevent the integral from accumulating to infinity.

Let's begin! Open [learning_1_gymnasium_pybullet_basic.py](file:///home/aiot/Projects/RocketLander/learning/learning_1_gymnasium_pybullet_basic.py) to start your first step!
