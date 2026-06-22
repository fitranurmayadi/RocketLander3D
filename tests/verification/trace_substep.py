import pybullet as p
import math
import numpy as np

p.connect(p.DIRECT)
urdf_path = "/home/aiot/Projects/RocketLander/rocket_lander/envs/assets/rocket_lander.urdf"
rocketId = p.loadURDF(urdf_path, [0, 0, 1000], p.getQuaternionFromEuler([0.7, 0, 0]))

p.setGravity(0, 0, -9.81)
dt = 1/240.0
p.setTimeStep(dt)

# Mock Action: Max Pitch Gimbal + Max Pitch RCS
force_mag = 3500.0
target_pitch = 0.35 # Rad
rcs_pitch = -1.0 # Negated sign as per my fix
r_torq = 1000.0 * 2.0

print(f"Sub-step Trace (Thrust=3500N, RCS=2000Nm):")
for i in range(8):
    # Main Engine
    tx = math.sin(target_pitch) * force_mag
    tz = math.cos(target_pitch) * force_mag
    p.applyExternalForce(rocketId, -1, [tx, 0, tz], [0, 0, -2.8], p.LINK_FRAME)
    
    # RCS
    p.applyExternalTorque(rocketId, -1, [0, rcs_pitch * r_torq, 0], p.LINK_FRAME)
    
    p.stepSimulation()
    _, v_ang = p.getBaseVelocity(rocketId)
    print(f"Sub-step {i}: AngVel_Y = {v_ang[1]:.4f} rad/s")

p.disconnect()
