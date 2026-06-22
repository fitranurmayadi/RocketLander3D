import gymnasium as gym
import rocket_lander
import pybullet as p
import numpy as np

def check_scales():
    env = gym.make("RocketLander-v0", render_mode=None)
    obs, _ = env.reset()
    
    print("\n--- SCALE DIAGNOSTICS ---")
    body_id = env.unwrapped.rocketId
    
    # Check Link -1 (base_link)
    res = p.getVisualShapeData(body_id)
    print(f"Number of visual shapes: {len(res)}")
    
    for shape in res:
        link_index = shape[1]
        mesh_path = shape[4].decode('utf-8')
        scale = shape[3]
        print(f"Link {link_index}: Mesh={mesh_path}, Scale={scale}")
        
    # Check AABB
    for i in range(-1, p.getNumJoints(body_id)):
        aabb_min, aabb_max = p.getAABB(body_id, i)
        size = np.array(aabb_max) - np.array(aabb_min)
        link_name = "base_link" if i == -1 else p.getJointInfo(body_id, i)[12].decode('utf-8')
        print(f"Link {i} ({link_name}) AABB Size: {size}")

    p.disconnect()

if __name__ == "__main__":
    check_scales()
