import argparse
import math
import numpy as np
import pybullet as p
from rocket_lander.envs.rocket_lander_env import RocketLanderEnv
from rocketlander.mission.state_machine import MissionStateMachine, MissionConfig, MissionPhase, RocketState
from rocketlander.trajectory.planner import TrajectoryPlanner
from rocketlander.guidance.guidance import GuidanceModule
from rocketlander.controllers.altitude_controller import AltitudeController
from rocketlander.controllers.horizontal_controller import HorizontalController
from rocketlander.controllers.attitude_controller import AttitudeController
from rocketlander.controllers.actuator_mixer import ActuatorMixer
from rocketlander.telemetry.telemetry import TelemetryLogger
from rocketlander.visualization.report import generate_report

def main():
    parser = argparse.ArgumentParser(description="RocketLander3D Mission Simulator")
    parser.add_argument("--no-render", action="store_true", help="Disable GUI")
    parser.add_argument("--ascent-alt", type=float, default=150.0)
    parser.add_argument("--waypoint", type=float, nargs=3, default=[500.0, 500.0, 1000.0])
    parser.add_argument("--max-steps", type=int, default=30000)
    parser.add_argument("--record", type=str, default="", help="Path to save mp4 video")
    args = parser.parse_args()

    env_kwargs = {
        "render_mode": "rgb_array" if args.record else (None if args.no_render else "human"),
        "normalize_obs": False,
        "randomize_spawn": False,
        "mission_mode": True
    }
    
    env = RocketLanderEnv(**env_kwargs)
    
    # Configure mission
    config = MissionConfig()
    config.ascent_target_alt = args.ascent_alt
    config.waypoint = args.waypoint
    
    # Components
    sm = MissionStateMachine(config)
    planner = TrajectoryPlanner(config)
    guidance = GuidanceModule(planner)
    
    # The environment runs at 60Hz control (dt = 4/240 = 1/60)
    dt = 1.0 / 60.0
    alt_ctrl = AltitudeController(dt)
    hor_ctrl = HorizontalController(dt)
    att_ctrl = AttitudeController(dt)
    mixer = ActuatorMixer()
    telemetry = TelemetryLogger()

    # Reset environment at ground level
    options = {
        "initial_pos": [0.0, 0.0, 3.0], # Legs will touch ground
        "initial_orn": p.getQuaternionFromEuler([0, 0, 0]),
        "initial_vel": [0.0, 0.0, 0.0],
        "initial_ang_vel": [0.0, 0.0, 0.0]
    }
    obs, info = env.reset(options=options)
    
    # Calculate full trajectory
    planner.plan_full_mission(options["initial_pos"], options["initial_vel"])
    
    state = RocketState()
    mission_time = 0.0
    
    print("Launch pad pre-launch sequence started...")
    
    import cv2
    video_writer = None
    if args.record:
        # RocketLander env returns 480x640 rgb array
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(args.record, fourcc, 60.0, (640, 480))

    prev_des_r = 0.0
    prev_des_p = 0.0

    try:
        for step in range(args.max_steps):
            if env.render_mode is not None:
                frame = env.render()
                if args.record and frame is not None:
                    # Convert RGB to BGR for cv2
                    video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            # 1. Parse observation to state
            state.pos = obs[0:3]
            state.orn_quat = obs[3:7]
            state.orn_euler = p.getEulerFromQuaternion(state.orn_quat)
            state.vel = obs[7:10]
            state.ang_vel = obs[10:13]
            state.contacts = obs[13:17]
            state.fuel = obs[18]
            state.t = mission_time
            
            # 2. Update state machine
            phase = sm.update(state)
            
            if sm.is_terminal:
                print(f"Mission ended in phase: {phase.name}")
                break

            # 3. Guidance (only advance trajectory time after PRELAUNCH)
            traj_time = mission_time - config.ignition_delay if phase != MissionPhase.PRELAUNCH else 0.0
            traj_time = max(0.0, traj_time)
            
            cmd = guidance.compute(state.pos, state.vel, state.orn_euler, traj_time, phase)

            # 4. Control Logic per Phase
            throttle = 0.0
            gimbal_p = 0.0
            gimbal_r = 0.0
            u_roll = 0.0
            u_pitch = 0.0
            u_yaw = 0.0
            des_r = 0.0
            des_p = 0.0
            legs_deploy = True if state.pos[2] < 50.0 else False

            if phase == MissionPhase.PRELAUNCH:
                throttle = 0.0
                
            elif phase == MissionPhase.ASCENT:
                throttle = alt_ctrl.compute(cmd.pos_error[2], cmd.vel_error[2], cmd.desired_acc[2], current_alt=state.pos[2])
                des_p, des_r, _ = hor_ctrl.compute(cmd.pos_error, cmd.vel_error, cmd.desired_acc, state.orn_euler[2], phase)
                # Clamp tilt to only go forward towards the waypoint (no backward tilt)
                des_p = max(0.0, des_p)
                des_r = min(0.0, des_r)

            elif phase == MissionPhase.WAYPOINT_NAV:
                throttle = alt_ctrl.compute(cmd.pos_error[2], cmd.vel_error[2], cmd.desired_acc[2], current_alt=state.pos[2])
                des_p, des_r, _ = hor_ctrl.compute(cmd.pos_error, cmd.vel_error, cmd.desired_acc, state.orn_euler[2], phase)
                # Clamp tilt to only go forward towards the waypoint (no backward tilt)
                des_p = max(0.0, des_p)
                des_r = min(0.0, des_r)

            elif phase == MissionPhase.BOOSTBACK:
                throttle = alt_ctrl.compute(cmd.pos_error[2], cmd.vel_error[2], cmd.desired_acc[2], current_alt=state.pos[2])
                des_p, des_r, _ = hor_ctrl.compute(cmd.pos_error, cmd.vel_error, cmd.desired_acc, state.orn_euler[2], phase)

            elif phase in [MissionPhase.ENTRY_BURN, MissionPhase.LANDING_BURN]:
                throttle = alt_ctrl.compute(cmd.pos_error[2], cmd.vel_error[2], cmd.desired_acc[2], current_alt=state.pos[2])
                des_p, des_r, _ = hor_ctrl.compute(cmd.pos_error, cmd.vel_error, cmd.desired_acc, state.orn_euler[2], phase)

                    
            # Rate limit the desired attitude commands to prevent high angular velocity and jagged steps
            # Limit rate of change to 0.8 rad/s (~46 deg/s)
            max_rate = 0.8  # rad/s
            max_step = max_rate * dt
            des_r = max(prev_des_r - max_step, min(prev_des_r + max_step, des_r))
            des_p = max(prev_des_p - max_step, min(prev_des_p + max_step, des_p))
            prev_des_r = des_r
            prev_des_p = des_p

            # Scale yaw error by cos(pitch) to compensate for the 1/cos(pitch) Euler rate singularity, keeping yaw stable.
            # Roll and Pitch remain unscaled to preserve perfect symmetry and control authority.
            roll_err = des_r - state.orn_euler[0]
            pitch_err = des_p - state.orn_euler[1]
            yaw_err = (cmd.desired_heading - state.orn_euler[2]) * math.cos(state.orn_euler[1])
            
            u_roll, u_pitch, u_yaw = att_ctrl.compute(roll_err, pitch_err, yaw_err, state.ang_vel)

            # Apply tilt compensation to throttle so rocket doesn't lose altitude when tilted
            # Approximating cosine of total tilt
            tilt_cos = math.cos(state.orn_euler[0]) * math.cos(state.orn_euler[1])
            throttle = throttle / max(0.2, tilt_cos)
            throttle = max(0.0, min(1.0, throttle))

            # Disable gimbal, rely on RCS globally (USER request)
            gimbal_p = 0.0
            gimbal_r = 0.0

            # 5. Mixer
            action = mixer.mix(throttle, gimbal_p, gimbal_r, u_roll, u_pitch, u_yaw, legs_deploy)
            
            # 6. Step env
            obs, reward, terminated, truncated, info = env.step(action)
            mission_time += dt

            # 7. Telemetry
            # Calculate mathematically pure planned attitude (Feed-Forward only) for symmetric plotting
            pure_ax, pure_ay = cmd.desired_acc[0], cmd.desired_acc[1]
            cy, sy = math.cos(cmd.desired_heading), math.sin(cmd.desired_heading)
            pure_bx = pure_ax * cy + pure_ay * sy
            pure_by = -pure_ax * sy + pure_ay * cy
            planned_p = math.atan2(pure_bx, 9.81)
            planned_r = math.atan2(-pure_by, 9.81)
            
            # Apply the identical directional constraint to the 'planned' telemetry so the graph reflects our intent
            if phase in [MissionPhase.ASCENT, MissionPhase.WAYPOINT_NAV]:
                planned_p = max(0.0, planned_p)
                planned_r = min(0.0, planned_r)
            
            pad_dist = math.sqrt(state.pos[0]**2 + state.pos[1]**2)
            telemetry.log(mission_time, phase, state.pos, state.vel, state.orn_euler, state.ang_vel, throttle, u_roll, u_pitch, u_yaw, cmd.desired_pos, planned_r, planned_p, cmd.desired_heading)
            
            if step % 10 == 0:
                telemetry.print_hud(mission_time, phase, state.pos, state.vel, state.orn_euler, throttle)

            if terminated or truncated:
                print(f"Env terminated. Phase: {phase.name}, Truncated: {truncated}")
                break

    finally:
        print("Mission complete. Generating report...")
        data = telemetry.get_data()
        generate_report(data, "images/ultimate_mission_report_rocketlander3d.png")
        env.close()
        if video_writer:
            video_writer.release()

if __name__ == "__main__":
    main()
