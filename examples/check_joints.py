import gymnasium as gym
import rocket_lander
import pybullet as p

def check_joints():
    env = gym.make("RocketLander-v0")
    env.reset()
    for i in range(p.getNumJoints(env.unwrapped.rocketId)):
        info = p.getJointInfo(env.unwrapped.rocketId, i)
        print(f"Joint {i}: {info[1].decode('utf-8')}")
    env.close()

if __name__ == "__main__":
    check_joints()
