import gymnasium as gym
import rocket_lander
import pybullet as p
import numpy as np

def check_com():
    env = gym.make("RocketLander-v0", render_mode=None)
    env.reset()
    
    body_id = env.unwrapped.rocketId
    
    print("\n--- LINK/COM DIAGNOSTICS ---")
    for i in range(-1, p.getNumJoints(body_id)):
        info = p.getDynamicsInfo(body_id, i)
        mass = info[0]
        local_inertia = info[2]
        com_pos = info[3]
        link_name = "base_link" if i == -1 else p.getJointInfo(body_id, i)[12].decode('utf-8')
        print(f"Link {i} ({link_name}): Mass={mass}, CoM={com_pos}")
        
    # Total Mass Check
    total_mass = 0
    for i in range(-1, p.getNumJoints(body_id)):
        total_mass += p.getDynamicsInfo(body_id, i)[0]
    print(f"Total Multibody Mass: {total_mass}")

    p.disconnect()

if __name__ == "__main__":
    check_com()
