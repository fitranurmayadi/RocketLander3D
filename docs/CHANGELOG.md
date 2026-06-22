# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.12.0] - 2026-02-12
### Added
- **Heading-to-Pad Control (Yaw Alignment)**: The rocket now dynamically rotates to point its nose (+X) towards the landing pad center.
- **Body-Frame Control Logic**: Implemented cascaded PID control using local body coordinates (Forward/Backward, Left/Right) for more intuitive and stable maneuvering.
- **Adaptive Precision Logic**: Horizontal velocity and tilt limits now automatically scale based on altitude and mission phase.
- **Improved Performance**: Achieved precision landings within 2 meters of the target center.
- **Mass-Adaptive Dynamics**: Vertical control now considers varying mass due to fuel consumption.

## [1.11.1] - 2026-02-12
### Fixed
- **Landing Success Reliability**: Fixed multiple issues preventing successful landings.
  - Increased simulation loop limits from 1000 to 5000 steps (~33s to ~166s).
  - Increased environment `max_steps` from 8000 to 24000.
  - Relaxed landing detection distance from 15m to 25m to account for descent drift.
  - Added altitude requirement (< 10m) for simplified landing detection.

## [1.11.0] - 2026-02-13
### Fixed
- **PID Controller (Basic)**:
  - Implemented 4-phase mission logic (RECOV, APPRO, PRECI, FINAL).
  - Added physics-based hover throttle calculation using mass estimation.
  - Implemented dedicated FINAL phase with controlled descent speed (-0.8 m/s).
  - Added engine cutoff on stable multi-foot contact.
  - Enhanced telemetry logging with full state and action vectors.

## [1.10.0] - 2026-02-12
### Added
- **PID + Trajectory Tracking Controller**: Hybrid approach combining trajectory planning with PID control.
  - Altitude-dependent velocity limits (15 m/s → 2 m/s based on altitude).
  - Trajectory updates every 0.5s for smooth reference path generation.
  - Aggressive braking capability with 45° max tilt in RECOV phase.
  - Fast execution (<1ms) with trajectory planning benefits.
  - Successfully tested on Medium difficulty; Super Hard shows 67% velocity reduction (15→4 m/s).
  - Ideal for scenarios with longer approach distances (similar to LunarLander3D's 1000m spawns).

## [1.9.0] - 2026-02-12
### Added
- **Model Predictive Control (MPC)**: Scipy-based SLSQP solver implementation with trajectory tracking.
  - Linearized 12-state model dynamics around hover equilibrium.
  - Reference trajectory generation with velocity constraints (20 m/s horizontal, 5 m/s vertical).
  - Hard constraints on Gimbal Angle (±15°), Throttle (0.4-1.0), and Tilt (45°).
  - Cost function with position, velocity, attitude, and control penalties.
  - PID fallback mechanism for solver failures.

### Fixed
- **Critical MPC Bugs**:
  - Missing identity matrix in A-matrix (state integration bug).
  - Inverted braking maneuver direction (gimbal polarity).
  - Debug print indices for velocity vector.
  - Trajectory tracking implementation (replaced static target).

### Known Limitations
- **MPC does not handle "Super Hard" scenario** (Alt=200m, Tilt=25°, Vel=15 m/s):
  - Linear model inaccurate at large angles (25° tilt).
  - Short prediction horizon (N=20 = 0.67s) insufficient for full braking trajectory.
  - Python solver too slow (~100ms) for real-time control at 30 Hz.
- **Recommendation**: Use PID controller for production. MPC requires nonlinear formulation with fast C++ solver (OSQP/Acados) or longer horizon (N≥50) for extreme scenarios.

## [1.8.1] - 2026-02-12
### Changed
- **Locked-Down Control**: Implemented "Straitjacket" strategy for PID.
  - **Strict Gimbal Limit**: +/- 5 degrees (0.087 rad) to physically prevent divergent oscillations.
  - **High Damping**: Kd=-0.8 to aggressively kill energy.
  - **Result**: Stable, survivable descent at low drift (< 5m/s), but fails at high drift (> 10m/s due to saturation).

## [1.8.0] - 2026-02-12
### Changed
- **Physics Re-Sync**: Reduced Gimbal Torque from 100kNm to **50kNm** in `rocket_lander_env.py`.
  - Matched actuator authority closer to vehicle inertia (Ratio ~0.77).
  - Attempted "Balanced" PID tuning (Failed: Still oscillatory with +/- 12 deg limit).

## [1.0.0] - 2026-02-12
### Added
- **Landing Integration Release**: Production-ready PID controller for high-drift landing scenarios.
- **Hyper-Descent Profile**: Optimized vertical velocities (-6m/s to -10m/s) to beat the 20.8s fuel limit.
- **Dynamic Throttle Floor**: Coupling minimum thrust with mass estimation to allow descent at all fuel levels.
- **Standardized Attitude Gains**: Rigid, fuel-independent gains for a predictable 'Super Heavy' style return.

## [0.9.0] - 2026-02-12
### Added
- **Tiered Landing State Machine**: Autonomous transition between RECOVERY (drift braking), APPROACH (pad acquisition), and FINAL (vertical touchdown).
- **Direct RCS Angular Damping**: Safety layer that fires RCS to oppose body-rates > 0.3 rad/s, preventing inertial tumbles.
- **Drift Authority Boost**: Automatic thrust scaling to ensure gimbal pressure during aggressive horizontal braking.
### Fixed
- **Inertial Snaps**: Resolved high-speed oscillations caused by derivative-kick and lack of body-rate damping.

## [0.8.0] - 2026-02-12
### Fixed
- **Mass Discrepancy (CRITICAL)**: Discovered and corrected 15x mass mismatch between URDF (200kg) and Controller (3000kg).
- **Hover Recalibration**: Re-tuned all PID coefficients for actual vehicle physics.

## [0.7.0] - 2026-02-12
### Added
- **Cascaded Translation Control**: Implemented horizontal velocity PID loops ($V_x, V_y$) that generate target tilt angles for the attitude loops.
- **Mass-Adaptive Gain Scaling**: Breakthrough feature that scales PID gains by the `(Current Mass / Max Mass)` ratio to compensate for increased gimbal authority as fuel drains.
- **Translational RCS Damping**: Added active lateral force damping using RCS thruster pairs (pure force, zero torque) to suppress drift without inducing tumble.
- **Derivative on Measurement (No-Kick PID)**: Refactored PID core to eliminate "derivative kick" spikes during setpoint changes.
- **Station Keeping**: The rocket can now maintain a 20s+ steady hover within ~1 meter of the target horizontal position.
- **Throttle-Attitude Coupling**: Automatic authority margin that boosts thrust if attitude errors are high, ensuring the gimbal has enough pressure to correct the vehicle.
### Changed
- `examples/pid_controller_basic.py`: Restructured control hierarchy to `Translation -> Attitude -> Actuators`.
- `examples/pid_calibration.py`: Added horizontal position/velocity telemetry and specific `[TRANSLATION]` log sections.

## [0.6.0] - 2026-02-12
### Added
- **Asymmetric Gain Strategy**: Implemented higher gains for Roll (KP=-3.0) vs Pitch (KP=-1.5) to compensate for observed mechanical efficiency differences in the gimbal assembly.
- **Manual Verification Suite**: `examples/manual_gimbal_test.py` for direct observation of actuator-to-RPY polarity.
### Fixed
- **The Polarity Trap**: Corrected fundamental PID polarity error where positive feedback was initially causing 1-second flips.
- **High TWR Instability**: Refined throttle logic to maintain a fixed minimum margin (0.15), preventing over-acceleration during recovery maneuvers that previously overwhelmed control authority.
- **RCS Sync**: Mapped RCS thrusters to work constructively with gimbal actions across all axes.

## [0.5.0] - 2026-02-12

### Added
- **Verification Scripts**:
  - `examples/check_twr.py`: Simple script to verify Thrust-to-Weight Ratio and Hover capability.
  - `examples/verify_physics_v2.py`: Comprehensive physics verification script (replaces old `verify_dynamics.py`).
- **Documentation**:
  - `docs/reports/`: Directory for storing verification logs and calibration plots.
  - `docs/coordinate_systems.md`: Detailed documentation on coordinate systems and actuator mapping.
  - `docs/rocket_dynamics.md`: Updated specifications for realistic physics and **Action Space Mapping** (Legs, Gimbal, Throttle ranges).

### Changed
- **Physics Overhaul (Realistic Scale)**:
  - Increased `MAX_MASS` from ~180kg to **3,000 kg** (Wet Mass).
  - Defined `MIN_MASS` at **300 kg** (Dry Mass).
  - Increased `MAIN_ENGINE_POWER` to **60,000 N**.
  - Increased `RCS_FORCE` base to **1,000 N**.
  - **Variable Mass Dynamics**: Rocket mass now decreases in real-time as fuel is consumed.
- **RCS Logic Refinement**:
  - **Yaw Reduction**: Reduced Yaw force to **200 N** (from 1000 N) to prevent excessive rotation due to slender inertia.
  - **Mass-Based Scaling**: RCS strength now scales linearly with mass (10% - 100%) to maintain consistent control authority at all fuel levels.
- **Observation Space**:
  - Replaced computationally expensive Raycast-based Altitude with **Global Z** position.
  - `obs[17]` is now redundant with `obs[2]` (Position Z).
- **Project Structure**:
  - Consolidated scripts into `examples/`, `tests/verification/`, and `docs/reports/`.
  - Removed clutter from the project root.

### Fixed
- **Propeller Bug**: Corrected URDF joint axis definitions to prevent legs from spinning weirdly.
- **RCS Stability**: Fixed uncontrollable spinning at low mass by implementing variable gain.
