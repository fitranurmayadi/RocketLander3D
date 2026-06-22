#!/usr/bin/env python3
"""Minimal test: spawn rocket low, let it drop, check if landing is detected."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import gymnasium as gym
import rocket_lander
import numpy as np
import pybullet as p
import math

env = gym.make('RocketLander-v0', render_mode=None)
obs, _ = env.reset()

# Override spawn: place rocket at low altitude, vertical, near pad
# Directly set rocket position
p.resetBasePositionAndOrientation(
    env.unwrapped.rocketId,
    [0, 0, 8],  # 8m altitude, centered on pad
    [0, 0, 0, 1]  # Upright
)
p.resetBaseVelocity(env.unwrapped.rocketId, [0, 0, -1], [0, 0, 0])

print("=== Landing Detection Test ===")
print("Spawned at Alt=8m, center of pad, descending at 1 m/s")
print()

for i in range(2000):
    action = np.zeros(16)
    # Gentle thrust to slow descent
    if obs[2] > 5.0:
        action[0] = -0.2  # Low thrust
    else:
        action[0] = -1.0  # Engine off for final
    action[15] = 1.0  # Legs deployed
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    contacts = obs[13:17]
    rpy = p.getEulerFromQuaternion(obs[3:7])
    speed = np.linalg.norm(obs[7:10])
    dist = np.linalg.norm(obs[0:2])
    tilt = np.linalg.norm([rpy[0], rpy[1]])
    
    if i % 120 == 0 or terminated or truncated:
        print(f"Step {i:4d} | Alt={obs[2]:6.2f} Speed={speed:5.2f} Tilt={math.degrees(tilt):5.1f}° "
              f"Dist={dist:5.1f} Contacts={sum(contacts):.0f} Reward={reward:7.1f} "
              f"Term={terminated} Trunc={truncated}")
    
    if terminated or truncated:
        if reward > 0:
            print("\n🚀 SUCCESSFUL LANDING DETECTED!")
        else:
            print("\n💥 CRASH/TIMEOUT DETECTED")
        break

env.close()
print("\nDone.")
