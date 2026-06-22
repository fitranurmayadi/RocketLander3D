import gymnasium as gym
import rocket_lander
import pybullet as p
import numpy as np

def lean_test_v2():
    def test_axis(p_act, r_act, name):
        env = gym.make("RocketLander-v0")
        env.unwrapped.randomize_spawn = False # CRITICAL
        env.reset(options={"initial_pos": [0,0,20.0], "initial_orn": [0,0,0,1], "initial_vel": [0,0,0]})
        act = np.zeros(16) # NEUTRAL
        act[3:15] = -1.0 # RCS OFF
        act[0] = 0.5     # Throttle
        act[1] = p_act   # Pitch
        act[2] = r_act   # Roll
        
        for _ in range(10):
            env.step(act)
            
        v = p.getBaseVelocity(env.unwrapped.rocketId)[0]
        print(f"Test {name}: P={p_act}, R={r_act} -> Vel: [{v[0]:.4f}, {v[1]:.4f}]")
        env.close()

    test_axis(0.5, 0.0, "Pitch(+)") # Expect +X
    test_axis(-0.5, 0.0, "Pitch(-)") # Expect -X
    test_axis(0.0, 0.5, "Roll(+)")  # Expect ?
    test_axis(0.0, -0.5, "Roll(-)") # Expect ?

if __name__ == "__main__":
    lean_test_v2()
