import pybullet as p
import time
import numpy as np

p.connect(p.DIRECT)
urdf_path = "/home/aiot/Projects/RocketLander/rocket_lander/envs/assets/rocket_lander.urdf"
rocketId = p.loadURDF(urdf_path, [0, 0, 10])

# Disable Gravity
p.setGravity(0, 0, 0)

print("--- THRUST DIRECTION TEST ---")
pos_val, _ = p.getBasePositionAndOrientation(rocketId)
print(f"Initial Z: {pos_val[2]:.4f}")

# Apply Positive Z Force in LINK_FRAME
p.applyExternalForce(rocketId, -1, [0, 0, 1000], [0, 0, 0], p.LINK_FRAME)
for _ in range(10):
    p.stepSimulation()

pos_new, _ = p.getBasePositionAndOrientation(rocketId)
print(f"Z after [+1000] Force: {pos_new[2]:.4f}")
if pos_new[2] > pos_val[2]:
    print("RESULT: Positive Z Force in LINK_FRAME is UP.")
else:
    print("RESULT: Positive Z Force in LINK_FRAME is DOWN.")

# Test with gravity
print("\n--- TWR TEST (180kg, 3500N thrust) ---")
p.setGravity(0, 0, -9.81)
p.resetBasePositionAndOrientation(rocketId, [0, 0, 10], [0, 0, 0, 1])
pos_v, _ = p.getBasePositionAndOrientation(rocketId)

# Apply 3500N
for _ in range(60): # 1/4 second at default 240Hz
    p.applyExternalForce(rocketId, -1, [0, 0, 3500], [0, 0, 0], p.LINK_FRAME)
    p.stepSimulation()

pos_n, _ = p.getBasePositionAndOrientation(rocketId)
print(f"Z after 3500N Thrust (0.25s): {pos_n[2]:.4f}")
if pos_n[2] > pos_v[2]:
    print("RESULT: TWR > 1.0 (Rocket is climbing).")
else:
    print("RESULT: TWR < 1.0 (Rocket is falling).")

p.disconnect()
