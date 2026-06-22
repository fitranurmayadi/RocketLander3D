import gymnasium as gym
import rocket_lander
import pybullet as p
import numpy as np
import time

def verify_directional_authority():
    env = gym.make("RocketLander-v0", normalize_obs=False)
    
    def test_action(act_idx, val, name):
        env.reset(options={"initial_pos": [0,0,20.0], "initial_orn": [0,0,0,1]})
        act = np.full(16, -1.0)
        act[0] = 0.5 # 75% throttle to stay aloft
        act[act_idx] = val
        
        v_start = np.array(p.getBaseVelocity(env.unwrapped.rocketId)[0])
        for _ in range(20):
            env.step(act)
        v_end = np.array(p.getBaseVelocity(env.unwrapped.rocketId)[0])
        dv = v_end - v_start
        print(f"Action[{act_idx}]={val} ({name}) -> Vel Delta: [{dv[0]:.3f}, {dv[1]:.3f}, {dv[2]:.3f}]")

    print("--- ACTUATOR DIRECTIONAL TEST ---")
    test_action(1, 1.0, "Gimbal P (+)")
    test_action(1, -1.0, "Gimbal P (-)")
    test_action(2, 1.0, "Gimbal R (+)")
    test_action(2, -1.0, "Gimbal R (-)")
    env.close()

if __name__ == "__main__":
    verify_directional_authority()
