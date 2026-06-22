import gymnasium as gym
import rocket_lander
import pybullet as p
import numpy as np
import time
import sys

def verify_visual():
    print("\n========================================")
    print("   ROCKET DYNAMICS VISUAL VERIFICATION  ")
    print("========================================")
    sys.stdout.flush()
    
    # Enable Human Rendering
    env = gym.make("RocketLander-v0", render_mode="human")
    
    def reset_and_stabilize(height=50.0):
        # Reset at a visible height
        env.reset(options={"initial_pos": [0,0,height], "initial_orn": [0,0,0,1]})
        neutral = np.full(16, -1.0)
        neutral[1] = 0.0 # Gimbal P Center
        neutral[2] = 0.0 # Gimbal R Center
        
        # Slow down for visual feedback
        for _ in range(30): 
            env.step(neutral)
            time.sleep(1.0/60.0) 
        return neutral

    def eval_action_visual(name, action, duration=2.0):
        print(f"\n[VISUAL TEST] {name}")
        sys.stdout.flush()
        
        reset_and_stabilize()
        
        # Run Action with visual pacing
        steps = int(duration * 30)
        for i in range(steps):
            env.step(action)
            # No need for extra sleep if env.render() handles it, 
            # but human mode in this env has a sleep(dt) in render()
            # which is called inside step() if render_mode="human"
            
        time.sleep(0.5) # Pause after each test to see the result
        
    neutral = np.full(16, -1.0); neutral[1]=0; neutral[2]=0

    # 1. GRAVITY
    eval_action_visual("Gravity (Freefall)", neutral, duration=2.0)

    # 2. MAIN ENGINE
    thrust_act = neutral.copy(); thrust_act[0] = 0.2 # low thrust for visual
    eval_action_visual("Main Engine (Low Thrust)", thrust_act, duration=3.0)

    # 3. PITCH (RCS)
    act_pitch_up = neutral.copy(); act_pitch_up[4] = 1.0; act_pitch_up[7] = 1.0
    eval_action_visual("RCS Pitch Up (+)", act_pitch_up, duration=1.5)
    
    act_pitch_down = neutral.copy(); act_pitch_down[3] = 1.0; act_pitch_down[8] = 1.0
    eval_action_visual("RCS Pitch Down (-)", act_pitch_down, duration=1.5)

    # 4. YAW (RCS)
    act_yaw_left = neutral.copy(); act_yaw_left[11] = 1.0; act_yaw_left[12] = 1.0
    eval_action_visual("RCS Yaw Left (+)", act_yaw_left, duration=1.5)

    # 5. GIMBAL PITCH
    act_gimbal_p = neutral.copy(); act_gimbal_p[0]=0.1; act_gimbal_p[1] = 1.0
    eval_action_visual("Gimbal Pitch Action", act_gimbal_p, duration=3.0)

    # 6. LANDING SENSOR & LEGS
    print("\n[VISUAL TEST] Landing Sensor & Legs")
    env.reset(options={"initial_pos": [0,0,5.0], "initial_orn": [0,0,0,1]})
    drop_action = neutral.copy()
    drop_action[15] = 1.0 # Legs DEPLOY
    
    for _ in range(120):
        env.step(drop_action)
        
    time.sleep(2.0)
    env.close()
    print("\nVISUAL VERIFICATION COMPLETE")

if __name__ == "__main__":
    verify_visual()
