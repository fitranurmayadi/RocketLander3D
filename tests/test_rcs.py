import pybullet as p
import time
from rocket_lander.envs.rocket_lander_env import RocketLanderEnv
import numpy as np

env = RocketLanderEnv()
obs, _ = env.reset()

# Force rocket to perfectly upright, 0 velocity
p.resetBasePositionAndOrientation(env.rocketId, [0, 0, 150], [0, 0, 0, 1])
p.resetBaseVelocity(env.rocketId, [0, 0, 0], [0, 0, 0])

action = np.zeros(16)
action[3:15] = -1.0 # All RCS off
action[5] = 1.0 # rcs[2] full power

def get_euler(obs):
    # obs is updated after step, but we want to read it accurately
    _, orn = p.getBasePositionAndOrientation(env.rocketId)
    return p.getEulerFromQuaternion(orn)

print("Initial Euler:", get_euler(obs))
for i in range(10):
    obs, _, _, _, _ = env.step(action)
    print(f"Step {i} Euler:", get_euler(obs))
