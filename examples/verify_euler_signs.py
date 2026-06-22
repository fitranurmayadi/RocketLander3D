import gymnasium as gym
import rocket_lander
import pybullet as p
import numpy as np
import math

def verify_euler_signs():
    env = gym.make("RocketLander-v0")
    env.reset(options={"initial_pos": [0,0,10.0], "initial_orn": [0,0,0,1]})
    
    neutral = np.full(16, -1.0); neutral[1]=0; neutral[2]=0
    
    # Test Gimbal Pitch (+)
    env.reset(options={"initial_pos": [0,0,10.0], "initial_orn": [0,0,0,1]})
    act = neutral.copy(); act[0]=0.0; act[1]=1.0
    for _ in range(30): env.step(act)
    pos, quat = p.getBasePositionAndOrientation(env.unwrapped.rocketId)
    rpy = p.getEulerFromQuaternion(quat)
    print(f"Action[1]=+1.0 (Gimbal P) -> Pitch Euler: {rpy[1]:.3f} rad")

    # Test Gimbal Roll (+)
    env.reset(options={"initial_pos": [0,0,10.0], "initial_orn": [0,0,0,1]})
    act = neutral.copy(); act[0]=0.0; act[2]=1.0
    for _ in range(30): env.step(act)
    pos, quat = p.getBasePositionAndOrientation(env.unwrapped.rocketId)
    rpy = p.getEulerFromQuaternion(quat)
    print(f"Action[2]=+1.0 (Gimbal R) -> Roll Euler: {rpy[0]:.3f} rad")
    
    env.close()

if __name__ == "__main__":
    verify_euler_signs()
