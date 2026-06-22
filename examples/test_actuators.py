import gymnasium as gym
import time
import numpy as np
import pybullet as p
import rocket_lander

def test_actuators():
    # Create Environment
    env = gym.make("RocketLander-v0", render_mode="human")
    
    # Spawn at 0,0,100 Upright
    print("Spawning at (0, 0, 100) Upright...")
    initial_pos = [0, 0, 100]
    initial_orn = p.getQuaternionFromEuler([0, 0, 0])
    
    obs, info = env.reset(options={"initial_pos": initial_pos, "initial_orn": initial_orn})
    
    # Wait for settle
    for _ in range(60):
        env.step(np.zeros(7))
        time.sleep(1/60)
        
    # Helper to run action for N seconds
    def run_action(action, duration, desc):
        print(f"Testing: {desc}")
        steps = int(duration * 30)
        for _ in range(steps):
            env.step(action)
            time.sleep(1/30)
            
    # 1. Main Engine Throttle (50%)
    # Action[0] is throttle: -1=0%, 1=100%. So 0.0 is 50%.
    run_action(np.array([0.0, 0, 0, 0, 0, 0, -1]), 2.0, "Main Engine 50% Throttle")
    
    # 2. Main Engine Gimbal Pitch
    run_action(np.array([-1.0, 1.0, 0, 0, 0, 0, -1]), 1.0, "Gimbal Pitch Positive")
    run_action(np.array([-1.0, -1.0, 0, 0, 0, 0, -1]), 1.0, "Gimbal Pitch Negative")
    
    # 3. Main Engine Gimbal Roll
    run_action(np.array([-1.0, 0, 1.0, 0, 0, 0, -1]), 1.0, "Gimbal Roll Positive")
    run_action(np.array([-1.0, 0, -1.0, 0, 0, 0, -1]), 1.0, "Gimbal Roll Negative")
    
    # 4. RCS Yaw
    # Action[4] is Yaw
    run_action(np.array([-1.0, 0, 0, 0, 1.0, 0, -1]), 1.5, "RCS Yaw Left")
    run_action(np.array([-1.0, 0, 0, 0, -1.0, 0, -1]), 1.5, "RCS Yaw Right")

    # 5. RCS Pitch/Roll
    run_action(np.array([-1.0, 0, 0, 1.0, 0, 0, -1]), 1.0, "RCS Pitch Up")
    run_action(np.array([-1.0, 0, 0, 0, 0, 1.0, -1]), 1.0, "RCS Roll Right")
    
    # 6. Landing Legs
    # Action[6] > 0 deploy
    run_action(np.array([-1.0, 0, 0, 0, 0, 0, 1.0]), 2.0, "Deploying Legs")
    run_action(np.array([-1.0, 0, 0, 0, 0, 0, -1.0]), 2.0, "Retracting Legs")
    
    print("Test Complete.")
    time.sleep(2)
    env.close()

if __name__ == "__main__":
    test_actuators()
