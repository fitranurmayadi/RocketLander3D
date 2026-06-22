import pybullet as p
import time
import numpy as np

p.connect(p.DIRECT)
urdf_path = "/home/aiot/Projects/RocketLander/rocket_lander/envs/assets/rocket_lander.urdf"
rocketId = p.loadURDF(urdf_path, [0, 0, 10])

# 1. Test Pitch Polarity
print("--- PITCH POLARITY TEST ---")
# Set orientation to +10 degrees pitch (Tilted toward +X)
p.resetBasePositionAndOrientation(rocketId, [0, 0, 10], p.getQuaternionFromEuler([0, 0.2, 0]))
_, quat = p.getBasePositionAndOrientation(rocketId)
r, p_val, y = p.getEulerFromQuaternion(quat)
print(f"Initial Pitch: {p_val:.4f}")

# Apply Negative Y Torque
p.applyExternalTorque(rocketId, -1, [0, -1000, 0], p.LINK_FRAME)
p.stepSimulation()

_, quat = p.getBasePositionAndOrientation(rocketId)
r, p_new, y = p.getEulerFromQuaternion(quat)
print(f"Pitch after Negative Y Torque: {p_new:.4f}")
if p_new < p_val:
    print("RESULT: Negative Y Torque counters Positive Pitch.")
else:
    print("RESULT: Negative Y Torque increases Positive Pitch.")

# 2. Test Roll Polarity
print("\n--- ROLL POLARITY TEST ---")
p.resetBasePositionAndOrientation(rocketId, [0, 0, 10], p.getQuaternionFromEuler([0.2, 0, 0]))
_, quat = p.getBasePositionAndOrientation(rocketId)
r_val, p_v, y = p.getEulerFromQuaternion(quat)
print(f"Initial Roll: {r_val:.4f}")

# Apply Negative X Torque
p.applyExternalTorque(rocketId, -1, [-1000, 0, 0], p.LINK_FRAME)
p.stepSimulation()

_, quat = p.getBasePositionAndOrientation(rocketId)
r_new, p_v, y = p.getEulerFromQuaternion(quat)
print(f"Roll after Negative X Torque: {r_new:.4f}")
if r_new < r_val:
    print("RESULT: Negative X Torque counters Positive Roll.")
else:
    print("RESULT: Negative X Torque increases Positive Roll.")

p.disconnect()
