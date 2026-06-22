import pybullet as p
import pybullet_data
import time
import os

# Updated Path: Now in examples/, so go up one level then into rocket_lander/assets/
URDF_PATH = os.path.join(os.path.dirname(__file__), "../rocket_lander/assets/rocket_lander.urdf")

def main():
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)

    p.loadURDF("plane.urdf")

    if not os.path.exists(URDF_PATH):
        print(f"Error: {URDF_PATH} not found.")
        print(f"Current dir: {os.getcwd()}")
        return

    startPos = [0, 0, 5]
    startOrientation = p.getQuaternionFromEuler([0, 0, 0])
    rocketId = p.loadURDF(URDF_PATH, startPos, startOrientation)

    num_joints = p.getNumJoints(rocketId)
    
    # Store indices for specific groups
    leg_joints = []
    other_sliders = []
    
    # Create Master Slider for Legs
    # Range +/- 1.6 rad
    legs_master_slider = p.addUserDebugParameter("Legs Master", -1.6, 1.6, 0)

    print(f"Loaded robot with {num_joints} joints.")
    
    for i in range(num_joints):
        info = p.getJointInfo(rocketId, i)
        joint_name = info[1].decode('utf-8')
        joint_type = info[2]
        
        if joint_type == p.JOINT_REVOLUTE or joint_type == p.JOINT_PRISMATIC:
            # Check if this is a leg joint
            if "foot" in joint_name:
                leg_joints.append(i)
                continue # Skip adding individual slider
            
            # Otherwise add normal slider
            lower = info[8]
            upper = info[9]
            slider = p.addUserDebugParameter(joint_name, lower, upper, 0)
            other_sliders.append((slider, i))
            print(f"Added slider for: {joint_name}")

    while p.isConnected():
        # 1. Apply Master Leg Control
        target_leg_pos = p.readUserDebugParameter(legs_master_slider)
        for j_idx in leg_joints:
             p.setJointMotorControl2(rocketId, j_idx, p.POSITION_CONTROL, targetPosition=target_leg_pos)

        # 2. Apply Other Sliders (Gimbals etc)
        for slider, j_idx in other_sliders:
            target_pos = p.readUserDebugParameter(slider)
            p.setJointMotorControl2(rocketId, j_idx, p.POSITION_CONTROL, targetPosition=target_pos)

        p.stepSimulation()
        time.sleep(1./240.)

if __name__ == "__main__":
    main()
