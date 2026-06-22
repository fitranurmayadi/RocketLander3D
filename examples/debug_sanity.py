
print("DEBUG: 1. Start", flush=True)
import gymnasium as gym
print("DEBUG: 2. Gym Imported", flush=True)
import pybullet as p
print("DEBUG: 3. PyBullet Imported", flush=True)
import sys
import os

# Force local import
curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(curr_dir)
sys.path.insert(0, root_dir)
print(f"DEBUG: 4. Path inserted: {root_dir}", flush=True)

import rocket_lander
print("DEBUG: 5. RocketLander Imported", flush=True)

try:
    env = gym.make("RocketLander-v0", render_mode=None)
    print("DEBUG: 6. Env Made", flush=True)
    env.reset()
    print("DEBUG: 7. Env Reset", flush=True)
    env.close()
    print("DEBUG: 8. Done", flush=True)
except Exception as e:
    print(f"ERROR: {e}", flush=True)
