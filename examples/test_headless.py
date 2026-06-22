import gymnasium as gym
import rocket_lander
import pybullet as p
import numpy as np

def test():
    print("Connecting to environment (Headless)...")
    env = gym.make("RocketLander-v0", render_mode=None)
    print("Resetting environment...")
    obs, info = env.reset()
    print(f"Initial Obs: {obs[0:3]}")
    print("Stepping environment...")
    for i in range(10):
        action = np.zeros(7)
        obs, reward, term, trunc, info = env.step(action)
        if i % 2 == 0: print(f"Step {i} completed. Z={obs[2]:.2f}")
    print("Test SUCCESSFUL.")
    env.close()

if __name__ == "__main__":
    test()
