from rocket_lander.envs.rocket_lander_env import RocketLanderEnv
import pybullet as p
import numpy as np

env = RocketLanderEnv()
options = {"initial_pos": [0, 0, 1000], "initial_orn": p.getQuaternionFromEuler([0.7, 0.7, 0.7])}
obs, info = env.reset(options=options)

# Mock Action from my previous failing log: Thr: 1.0, G_P: -1.0, G_R: 0.0, etc.
# Wait, my log showed Thr: 1.00 at step 35.
action = np.zeros(7)
action[0] = 1.0 # Max thrust
action[1] = -1.0 # Max pitch gimbal
action[3] = 1.0 # Max pitch RCS (Polarity was reversed in that failing run)

print(f"Beginning Live Step Debug...")
v_ang = obs[10:13]
print(f"Initial AngVel: {v_ang}")

# Manual Step simulation to see sub-steps
# (Replicating logic of RocketLanderEnv.step)

obs, reward, terminated, truncated, info = env.step(action)
print(f"Observation after 1 control step: {obs[10:13]}")
print(f"Terminated: {terminated}")
