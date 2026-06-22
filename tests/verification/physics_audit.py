import pybullet as p
import pybullet_data
import os

p.connect(p.DIRECT)
urdf_path = "/home/aiot/Projects/RocketLander/rocket_lander/envs/assets/rocket_lander.urdf"
rocketId = p.loadURDF(urdf_path)

print(f"--- PHYSICS AUDIT ---")
num_joints = p.getNumJoints(rocketId)
total_mass = 0
for i in range(-1, num_joints):
    dynamics = p.getDynamicsInfo(rocketId, i)
    mass = dynamics[0]
    total_mass += mass
    if i == -1:
        name = "base_link"
    else:
        name = p.getJointInfo(rocketId, i)[12].decode("utf-8")
    print(f"Link {i} ({name}): Mass={mass}, Inertia={dynamics[2]}")

print(f"\nTotal Multi-body Mass: {total_mass}")

# Check for unconstrained joints
for i in range(num_joints):
    joint_info = p.getJointInfo(rocketId, i)
    j_type = joint_info[2]
    j_name = joint_info[1].decode("utf-8")
    print(f"Joint {i} ({j_name}): Type={j_type}")

p.disconnect()
