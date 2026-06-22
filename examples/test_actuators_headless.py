import gymnasium as gym
import time
import numpy as np
import pybullet as p
import rocket_lander
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
file_handler = logging.FileHandler('actuator_test_results.log', mode='w')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

def test_actuators_headless():
    logger.info("Starting Headless Actuator Test (Reset Log)")
    
    # Create Environment
    env = gym.make("RocketLander-v0", render_mode=None)
    
    # Spawn at 0,0,100 Upright
    logger.info("Spawning at (0, 0, 100) Upright...")
    initial_pos = [0, 0, 100]
    initial_orn = p.getQuaternionFromEuler([0, 0, 0])
    
    obs, info = env.reset(options={"initial_pos": initial_pos, "initial_orn": initial_orn})
    
    # Helper to get state
    def get_state():
        pos, quat = p.getBasePositionAndOrientation(env.unwrapped.rocketId)
        lin_vel, ang_vel = p.getBaseVelocity(env.unwrapped.rocketId)
        return np.array(pos), np.array(lin_vel), np.array(ang_vel)

    # Helper to check joint position
    def get_joint_pos(joint_name):
        idx = env.unwrapped.joint_map.get(joint_name)
        if idx is not None:
             state = p.getJointState(env.unwrapped.rocketId, idx)
             return state[0] # Position
        return None

    # Wait for settle - use 0 throttle!
    logger.info("Waiting for physics to settle (Gravity only)...")
    zero_throttle_action = np.array([-1.0, 0, 0, 0, 0, 0, -1.0])
    
    start_pos_settle, _, _ = get_state()
    for _ in range(30):
        env.step(zero_throttle_action)
    
    start_pos, start_vel, start_ang = get_state()
    logger.info(f"State after settle: Z={start_pos[2]:.2f} (Delta: {start_pos[2]-start_pos_settle[2]:.2f}), Vz={start_vel[2]:.2f}")

    # ==========================
    # 1. Main Engine Verification
    # ==========================
    logger.info("--- Testing Main Engine (50% Thrust) ---")
    action = np.zeros(7)
    action[0] = 0.0 # 50% thrust
    action[6] = -1.0 # Retract legs
    
    start_vel_test = start_vel
    for _ in range(30): # 1 sec
        env.step(action)
        
    end_pos, end_vel, end_ang = get_state()
    accel_z = end_vel[2] - start_vel_test[2]
    logger.info(f"Main Engine Result: Vz changed from {start_vel_test[2]:.2f} to {end_vel[2]:.2f} (Delta: {accel_z:.2f})")
    
    # With TWR ~2.0, 50% thrust = 2000N. Gravity = 1765N. Net = +235N. Accel = +1.3 m/s^2.
    # Previous Vz should be negative (falling). New Vz should be less negative or positive.
    # Actually, we compare Accel.
    # Without thrust, Vz decreases by -9.8 m/s every second.
    # With thrust, Vz changed by `accel_z`.
    # Expected 'accel_z' due to gravity alone is -9.8.
    # If engine works, 'accel_z' should be > -9.8 (less negative or positive).
    
    if accel_z > -9.0: # Significant improvement over gravity
         logger.info("SUCCESS: Main Engine produced thrust (resisting gravity).")
    else:
         logger.warning("FAILURE: Main Engine thrust seems low.")

    # ==========================
    # 2. RCS Yaw Verification
    # ==========================
    logger.info("--- Testing RCS Yaw (rotate Left) ---")
    p.resetBaseVelocity(env.unwrapped.rocketId, [0,0,0], [0,0,0]) # Reset velocities
    
    action = np.zeros(7)
    action[0] = -1.0 # 0% thrust
    action[4] = 1.0 # Yaw positive
    
    for _ in range(30):
        env.step(action)
        
    _, _, yaw_vel = get_state()
    logger.info(f"RCS Yaw Result: Angular Velocity Z = {yaw_vel[2]:.4f}")
    if abs(yaw_vel[2]) > 0.05:
        logger.info("SUCCESS: RCS Yaw produced rotation.")
    else:
        logger.warning("FAILURE: RCS Yaw rotation negligible.")

    # ==========================
    # 3. Gimbal Verification
    # ==========================
    logger.info("--- Testing Gimbal Pitch ---")
    action = np.zeros(7)
    action[0] = -1.0
    action[1] = 1.0 # Gimbal Pitch Max
    
    for _ in range(10): 
        env.step(action)
        
    # Check joint angle directly
    gimbal_pitch_val = get_joint_pos("joint_rocket_thruster_main_engine_pitch")
    logger.info(f"Gimbal Pitch Joint Angle: {gimbal_pitch_val:.4f}")
    if abs(gimbal_pitch_val) > 0.05:
        logger.info("SUCCESS: Gimbal Pitch joint moved.")
    else:
        logger.warning("FAILURE: Gimbal Pitch joint did not move.")

    # ==========================
    # 4. Landing Legs Verification
    # ==========================
    logger.info("--- Testing Landing Legs Deployment ---")
    action = np.zeros(7)
    action[0] = -1.0
    action[6] = 1.0 # Deploy Legs
    
    for _ in range(60): # Give it time to move
        env.step(action)
        
    leg_joint_val = 0
    checked_leg = False
    for name, idx in env.unwrapped.joint_map.items():
        if "foot" in name and "joint" in name:
            leg_joint_val = get_joint_pos(name)
            logger.info(f"Leg Joint {name} Angle: {leg_joint_val:.4f}")
            checked_leg = True
            break
            
    if checked_leg and leg_joint_val > 0.5: # Target is 1.6
         logger.info("SUCCESS: Legs deployed.")
    elif checked_leg:
         logger.warning(f"FAILURE: Legs did not fully deploy. Angle: {leg_joint_val}")
    else:
         logger.warning("FAILURE: Could not find leg joints to verify.")

    logger.info("Test Complete. Results saved to actuator_test_results.log")
    env.close()

if __name__ == "__main__":
    test_actuators_headless()
