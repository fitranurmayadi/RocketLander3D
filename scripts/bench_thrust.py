import pybullet as p
import time
import math

p.connect(p.DIRECT)
urdf_path = "/home/aiot/Projects/RocketLander/rocket_lander/envs/assets/rocket_lander.urdf"
rocketId = p.loadURDF(urdf_path, [0, 0, 10])

# 3500N Thrust Up
force = 3500.0
dt = 1/240.0

print(f"Starting Benchmark (3500N Up, 180kg Mass)...")
pos_0, _ = p.getBasePositionAndOrientation(rocketId)
z0 = pos_0[2]

for i in range(240): # 1 second
    p.applyExternalForce(rocketId, -1, [0, 0, force], [0, 0, 0], p.LINK_FRAME)
    p.stepSimulation()

pos_1, _ = p.getBasePositionAndOrientation(rocketId)
z1 = pos_1[2]
vz = (z1 - z0) # Average delta-Z over 1s is v_avg. 
# z = v0*t + 1/2 * a * t^2. If v0=0, z = 1/2 * a * t^2. 
# a = 2 * delta_z / t^2 = 2 * (z1 - z0).
accel = 2 * (z1 - z0)

print(f"Altitude at 0s: {z0:.4f}")
print(f"Altitude at 1s: {z1:.4f}")
print(f"Calculated Acceleration: {accel:.4f} m/s^2")
print(f"Expected Acceleration: {(3500/180.1) - 9.81:.4f} m/s^2")

p.disconnect()
