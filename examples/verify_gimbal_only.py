import gymnasium as gym
import time
import numpy as np
import pybullet as p
import rocket_lander

def verify_gimbal_isolated():
    print("========================================")
    print("  ISOLATED GIMBAL STABILITY TEST        ")
    print("========================================")
    
    env = gym.make("RocketLander-v0", render_mode="human")
    
    def spawn_reset():
        opts = {"initial_pos": [0,0,50], "initial_orn": p.getQuaternionFromEuler([0,0,0])}
        return env.reset(options=opts)

    spawn_reset()
    
    # 1. HOLD IDLE
    print("Holding Idle for 2 seconds...")
    for _ in range(60):
        env.step(np.array([-1, 0, 0, 0, 0, 0, -1])) # Throttle 0, Gimbal 0
        time.sleep(1/30)

    # 2. STEP PITCH
    print("Stepping Pitch to 0.5 (Half Limit)...")
    for _ in range(90):
        env.step(np.array([-1, 0.5, 0, 0, 0, 0, -1]))
        time.sleep(1/30)

    # 3. HOLD PITCH
    print("Holding Pitch...")
    for _ in range(60):
        env.step(np.array([-1, 0.5, 0, 0, 0, 0, -1]))
        time.sleep(1/30)

    # 4. STEP ROLL
    print("Stepping Roll to 0.5...")
    for _ in range(90):
        env.step(np.array([-1, 0.5, 0.5, 0, 0, 0, -1]))
        time.sleep(1/30)

    # 5. OSCILLATION TEST
    print("Fast Sine Oscillation (Stress Test)...")
    for i in range(150):
        val = 0.5 * np.sin(i * 0.2)
        env.step(np.array([-1, val, val, 0, 0, 0, -1]))
        time.sleep(1/30)

    print("Test Complete.")
    env.close()

if __name__ == "__main__":
    verify_gimbal_isolated()
