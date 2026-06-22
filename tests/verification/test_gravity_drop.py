import pybullet as p
import pybullet_data
import time
import numpy as np

p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setTimeStep(1.0/240.0)

urdf_path = "/home/aiot/Projects/RocketLander/rocket_lander/envs/assets/rocket_lander.urdf"
rocketId = p.loadURDF(urdf_path, [0, 0, 10])

print("--- GRAVITY DROP TEST ---")
pos_val, _ = p.getBasePositionAndOrientation(rocketId)
print(f"Initial Z: {pos_val[2]:.4f}")

# Step for 1 second of simulation time
for _ in range(240):
    p.stepSimulation()

pos_new, _ = p.getBasePositionAndOrientation(rocketId)
vel, _ = p.getBaseVelocity(rocketId)
print(f"Z after 1.0s: {pos_new[2]:.4f}")
print(f"Vz after 1.0s: {vel[2]:.4f}")

# Expected: V = g * t = -9.81 * 1.0 = -9.81 m/s
# Expected: Z = Z0 + 0.5 * g * t^2 = 10 - 4.905 = 5.095 m

if abs(vel[2] + 9.81) < 0.1:
    print("RESULT: Gravity is working correctly (Vz ~ -9.81 m/s).")
else:
    print(f"RESULT: Gravity ERROR. Expected -9.81, got {vel[2]:.4f}")

p.disconnect()
