import gymnasium as gym
import rocket_lander
import pybullet as p
import numpy as np
import time
import math

def lean_test():
    env = gym.make("RocketLander-v0", render_mode=None)
    
    def test_lean(ax_target, ay_target, name):
        obs, _ = env.reset(options={"initial_pos": [0,0,20.0], "initial_orn": [0,0,0,1]})
        # Target [ax, ay] World. 
        # Yaw is 0, so Forward is +X, Left is +Y.
        
        # Test 1: Let's see if Action 1+ moves us to +X.
        act = np.full(16, -1.0)
        act[0] = 0.5 # Hover
        if ax_target > 0: act[1] = 0.5
        elif ax_target < 0: act[1] = -0.5
        if ay_target > 0: act[2] = -0.5
        elif ay_target < 0: act[2] = 0.5
        
        for _ in range(30):
            env.step(act)
        
        v = p.getBaseVelocity(env.unwrapped.rocketId)[0]
        print(f"Test {name}: Act[1]={act[1]}, Act[2]={act[2]} -> Vel: [{v[0]:.2f}, {v[1]:.2f}]")

    test_lean(1, 0, "+X (Forward)")
    test_lean(0, 1, "+Y (Left)")
    env.close()

if __name__ == "__main__":
    lean_test()
