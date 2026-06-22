import gymnasium as gym
import pybullet as p
import numpy as np
import math
import random

def setup_landing_scenario(env, difficulty="medium"):
    """
    Sets up a Sky-To-Ground landing scenario.
    Rocket starts at altitude with initial velocity and tilt.
    """
    if difficulty == "easy":
        alt = random.uniform(50, 70)
        tilt = random.uniform(-5, 5)
        vel_h = random.uniform(-2, 2)
    elif difficulty == "hard":
        alt = random.uniform(150, 300)
        tilt = random.uniform(-30, 30)
        vel_h = random.uniform(-20, 20)
    elif difficulty == "super_hard":
        alt = 200.0
        tilt = 25.0
        vel_h = 15.0 # Forced High Drift
    elif difficulty == "extreme":
        # LunarLander3D-style long distance
        alt = 1000.0
        tilt = 45.0
        vel_h = 15.0
    else: # medium
        alt = random.uniform(80, 150)
        tilt = random.uniform(-15, 15)
        vel_h = random.uniform(-10, 10)

    # For extreme difficulty, spawn further away horizontally too
    if difficulty == "extreme":
        initial_pos = [random.uniform(-100, 100), random.uniform(-100, 100), alt]
    else:
        initial_pos = [random.uniform(-30, 30), random.uniform(-30, 30), alt]
    initial_orn = p.getQuaternionFromEuler([0, math.radians(tilt), 0])
    
    obs, info = env.reset(options={
        "initial_pos": initial_pos,
        "initial_orn": initial_orn
    })
    
    # Apply initial velocities via PyBullet directly after reset
    p.resetBaseVelocity(env.unwrapped.rocketId, [vel_h, 0, -5.0], [0, 0, 0])
    
    print(f"Scenario [{difficulty.upper()}]: Alt={alt:.1f}m, Tilt={tilt:.1f}deg, Vel_H={vel_h:.1f}m/s")
    return obs

if __name__ == "__main__":
    import rocket_lander
    # Test execution
    env = gym.make("RocketLander-v0", render_mode="human")
    obs = setup_landing_scenario(env, difficulty="medium")
    for _ in range(100):
        env.step(np.zeros(16)) # Just fall
    env.close()
