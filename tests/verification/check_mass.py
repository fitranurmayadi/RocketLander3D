import pybullet as p
import pybullet_data
import os
import math

p.connect(p.DIRECT)
urdf_path = "/home/aiot/Projects/RocketLander/rocket_lander/envs/assets/rocket_lander.urdf"
rocketId = p.loadURDF(urdf_path, [0, 0, 10])

total_mass = 0
for i in range(-1, p.getNumJoints(rocketId)):
    if i == -1:
        m = p.getDynamicsInfo(rocketId, -1)[0]
    else:
        m = p.getDynamicsInfo(rocketId, i)[0]
    total_mass += m
    name = "base_link" if i == -1 else p.getJointInfo(rocketId, i)[12].decode("utf-8")
    print(f"Link {i} ({name}): Mass = {m}")

print(f"\nTOTAL MASS: {total_mass} kg")
print(f"Gravity Force: {total_mass * 9.81} N")
print(f"Required Thrust (TWR 2.0): {total_mass * 9.81 * 2.0} N")

p.disconnect()
