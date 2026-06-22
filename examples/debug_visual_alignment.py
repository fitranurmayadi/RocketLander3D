import pybullet as p
import pybullet_data
import time
import os
import math

URDF_PATH = "rocket_lander/envs/assets/rocket_lander.urdf"

def debug_visual():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    
    # Load with axes
    rocketId = p.loadURDF(URDF_PATH, [0, 0, 5], useFixedBase=True)
    
    num_joints = p.getNumJoints(rocketId)
    print(f"Joints: {num_joints}")
    
    # Add debug lines for each joint
    for i in range(-1, num_joints):
        # Base link (-1) or joint link
        name = "base_link" if i == -1 else p.getJointInfo(rocketId, i)[12].decode('utf-8')
        print(f"Visualizing: {name}")
        
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    p.resetDebugVisualizerCamera(cameraDistance=10, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0,0,5])

    while p.isConnected():
        # Draw axes at each link origin
        for i in range(-1, num_joints):
            state = p.getLinkState(rocketId, i) if i >= 0 else [[0,0,5], [0,0,0,1]]
            pos = state[0]
            orn = state[1]
            
            # Local Axes
            rot_mat = p.getMatrixFromQuaternion(orn)
            # Row major: [0,1,2] = X, [3,4,5] = Y, [6,7,8] = Z
            vec_x = [rot_mat[0], rot_mat[3], rot_mat[6]]
            vec_y = [rot_mat[1], rot_mat[4], rot_mat[7]]
            vec_z = [rot_mat[2], rot_mat[5], rot_mat[8]]
            
            length = 0.5
            p.addUserDebugLine(pos, [pos[0]+vec_x[0]*length, pos[1]+vec_x[1]*length, pos[2]+vec_x[2]*length], [1,0,0], 1, 0.1)
            p.addUserDebugLine(pos, [pos[0]+vec_y[0]*length, pos[1]+vec_y[1]*length, pos[2]+vec_y[2]*length], [0,1,0], 1, 0.1)
            p.addUserDebugLine(pos, [pos[0]+vec_z[0]*length, pos[1]+vec_z[1]*length, pos[2]+vec_z[2]*length], [0,0,1], 1, 0.1)
            
        time.sleep(0.1)
        p.removeAllUserDebugItems()

if __name__ == "__main__":
    debug_visual()
