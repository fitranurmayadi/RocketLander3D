import pybullet as p
import pybullet_data
import time
import numpy as np
import rocket_lander
import gymnasium as gym

def test_axis_authority():
    print("DEBUG: Importing rocket_lander package (Executing __init__.py)")
    env = gym.make("RocketLander-v0")
    env.reset()
    
    # Get ID
    try:
        rocketId = env.unwrapped.rocketId
    except:
        rocketId = env.rocketId
        
    p.setGravity(0, 0, -9.81)
    
    print(f"Base Link Mass: {p.getDynamicsInfo(rocketId, -1)[0]}")
    
    # Test 1: Pitch Axis
    # Result mapping: Gimbal Pitch -> Pitch Rate -> Pitch Angle -> X-Accel
    print("\n--- TEST 1: PITCH AXIS ---")
    env.reset()
    try:
        rocketId = env.unwrapped.rocketId
    except:
        rocketId = env.rocketId
        
    print(f"Rocket ID: {rocketId}")
    num_joints = p.getNumJoints(rocketId)
    print(f"Num Joints: {num_joints}")
    # Hardcoded based on Joint List:
    # Joint 0: joint_rocket_thruster_main_engine_pitch
    # Joint 1: joint_rocket_thruster_main_engine_roll
    gimbal_r_idx = 1
    gimbal_p_idx = 0
    
    print(f"Using Indices: Roll={gimbal_r_idx}, Pitch={gimbal_p_idx}")
    
    # Find Engine Link Index
    engine_link_idx = -1
    for i in range(num_joints):
        info = p.getJointInfo(rocketId, i)
        lname = info[12].decode('utf-8')
        if "engine" in lname and "main" in lname: 
            engine_link_idx = i
            print(f"Engine Link Found: {lname} (Index {i})")
            
    p.resetBasePositionAndOrientation(rocketId, [0, 0, 10], [0, 0, 0, 1])
    p.resetBaseVelocity(rocketId, [0, 0, 0], [0, 0, 0])
    
    print(f"Applying +0.2 rad to Joint {gimbal_p_idx} (Pitch)")
    
    for i in range(50):
        # Hold Gimbal
        p.setJointMotorControl2(rocketId, gimbal_r_idx, p.POSITION_CONTROL, targetPosition=0.0, force=50000)
        p.setJointMotorControl2(rocketId, gimbal_p_idx, p.POSITION_CONTROL, targetPosition=0.2, force=50000) # Pitch
        p.stepSimulation()
        
        # Apply Thrust Vectoring
        # Get Engine Link Orientation
        if engine_link_idx != -1:
            state = p.getLinkState(rocketId, engine_link_idx)
            e_orn = state[1] # Quaternion
            # Rotate [0,0,F] by e_orn
            force_mag = 60000.0
            f_world, _ = p.multiplyTransforms([0,0,0], e_orn, [0,0,force_mag], [0,0,0,1])
            # Apply to Base Link at Engine Pos? Or Center of Mass?
            # Env applies to -1 (Base) at e_pos.
            e_pos = state[0]
            p.applyExternalForce(rocketId, -1, f_world, e_pos, p.WORLD_FRAME)
        else:
            p.applyExternalForce(rocketId, -1, [0, 0, 60000], [0, 0, 0], p.LINK_FRAME)
            
        p.stepSimulation()
        
    # Measure
    pos, quat = p.getBasePositionAndOrientation(rocketId)
    lin_vel, ang_vel = p.getBaseVelocity(rocketId)
    rpy = p.getEulerFromQuaternion(quat)
    
    print(f"Result after 50 steps:")
    print(f"  Pitch Angle (Theta): {rpy[1]:.4f} rad")
    print(f"  Pitch Rate (Wy):     {ang_vel[1]:.4f} rad/s")
    print(f"  X-Velocity:          {lin_vel[0]:.4f} m/s")
    
    # Deductions
    print("Correlations:")
    if abs(rpy[1]) > 0.001:
        sign_g_to_theta = "+" if (rpy[1] > 0) else "-"
        print(f"  +GimbalPitch -> {sign_g_to_theta}PitchAngle")
    
    if abs(lin_vel[0]) > 0.01:
        sign_theta_to_vx = "+" if (np.sign(lin_vel[0]) == np.sign(rpy[1])) else "-"
        print(f"  +PitchAngle  -> {sign_theta_to_vx}X_Vel")


    # Test 3: RCS Yaw Authority
    print("\n--- TEST 3: RCS YAW ---")
    env.reset()
    p.resetBasePositionAndOrientation(rocketId, [0, 0, 10], [0, 0, 0, 1])
    p.resetBaseVelocity(rocketId, [0, 0, 0], [0, 0, 0])
    
    # Test Pair A: Config 8, 9 (Environment Indices) -> Action Indices 11, 12?
    # Config 0 is Action 3.
    # Config 8 is Action 11.
    # Config 9 is Action 12.
    
    # Apply +1.0 to Actions 11 and 12
    print("Activating Actions 11 & 12 (Config 8 & 9)")
    
    # We need to manually set RCS forces because Env Step consumes actions
    # Or we can just use the env step?
    # Let's use env step for simplicity.
    
    action = np.zeros(16)
    action[11] = 1.0
    action[12] = 1.0
    action[15] = -1.0 # Legs Retracted
    
    for i in range(10):
        obs, _, _, _, _ = env.step(action)
        
    # Measure
    # Obs 10-12 are ang vel
    wz = obs[12]
    print(f"Result (Actions 11, 12): Yaw Rate (Wz) = {wz:.4f} rad/s")
    
    # Test Pair B: Config 10, 11 -> Action 13, 14
    env.reset()
    p.resetBasePositionAndOrientation(rocketId, [0, 0, 10], [0, 0, 0, 1])
    p.resetBaseVelocity(rocketId, [0, 0, 0], [0, 0, 0])
    
    print("Activating Actions 13 & 14 (Config 10 & 11)")
    action = np.zeros(16)
    action[13] = 1.0
    action[14] = 1.0
    
    for i in range(10):
        obs, _, _, _, _ = env.step(action)
        
    wz = obs[12]
    print(f"Result (Actions 13, 14): Yaw Rate (Wz) = {wz:.4f} rad/s")
        
    # Measure
    pos, quat = p.getBasePositionAndOrientation(rocketId)
    lin_vel, ang_vel = p.getBaseVelocity(rocketId)
    rpy = p.getEulerFromQuaternion(quat)
    
    print(f"Result after 50 steps:")
    print(f"  Roll Angle (Phi):    {rpy[0]:.4f} rad")
    print(f"  Roll Rate (Wx):      {ang_vel[0]:.4f} rad/s")
    print(f"  Y-Velocity:          {lin_vel[1]:.4f} m/s")
    
    # Deductions
    print("Correlations:")
    if abs(rpy[0]) > 0.01:
        sign_g_to_phi = "+" if (rpy[0] > 0) else "-"
        print(f"  +GimbalRoll  -> {sign_g_to_phi}RollAngle")
    
    if abs(lin_vel[1]) > 0.1:
        sign_phi_to_vy = "+" if (np.sign(lin_vel[1]) == np.sign(rpy[0])) else "-"
        print(f"  +RollAngle   -> {sign_phi_to_vy}Y_Vel")
        
if __name__ == "__main__":
    test_axis_authority()
