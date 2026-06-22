import pybullet as p
import pybullet_data
import time
import numpy as np

p.connect(p.DIRECT)
p.setGravity(0, 0, 0)
p.setTimeStep(1.0/60.0)

urdf_path = "/home/aiot/Projects/RocketLander/rocket_lander/envs/assets/rocket_lander.urdf"
rocketId = p.loadURDF(urdf_path, [0, 0, 1000])

print("--- ACCELERATION TEST (Vacuum, 3500N, 180kg) ---")
# Apply 3500N for 60 steps (1.0 second)
for _ in range(60):
    p.applyExternalForce(rocketId, -1, [0, 0, 3500], [0, 0, 0], p.LINK_FRAME)
    p.stepSimulation()

vel, _ = p.getBaseVelocity(rocketId)
actual_a = vel[2] # Since t=1.0, v = a
expected_a = 3500.0 / 180.1
print(f"Expected Acceleration: {expected_a:.4f} m/s^2")
print(f"Actual Acceleration:   {actual_a:.4f} m/s^2")

if abs(actual_a - expected_a) < 0.1:
    print("RESULT: Mass/Thrust scaling is CORRECT.")
else:
    print(f"RESULT: Scaling ERROR. Factor = {actual_a / expected_a:.4f}x")

p.disconnect()
