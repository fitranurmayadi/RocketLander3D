import sys
import os

# Create path to the parent directory (project root)
# This ensures we can import rocket_lander even if not installed via pip
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

print(f"DEBUG: Added {parent_dir} to sys.path")

import gymnasium as gym
try:
    import rocket_lander
    from rocket_lander import RocketLanderEnv
    print("DEBUG: Import rocket_lander SUCCESS")
except ImportError as e:
    print(f"DEBUG: Import rocket_lander FAILED: {e}")
    sys.exit(1)

import pybullet as p
import time
import numpy as np

def main():
    # Initialize Environment
    # We use the class directly
    env = RocketLanderEnv(render_mode="human")
    obs, info = env.reset()
    
    # DISABLE DEFAULT KEYBOARD SHORTCUTS (W for Wireframe etc)
    p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0) # Optional: Hide Side panels
    p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)


    print("=== Rocket Lander Manual Control ===")
    print("Controls:")
    print("  W / S : Increase / Decrease Throttle")
    print("  Arrow Up / Down : Gimbal Pitch")
    print("  Arrow Left / Right : Gimbal Roll")
    print("  Q / E : RCS Yaw (Spin)")
    print("  G : Toggle Legs")
    print("  R : Reset")
    print("====================================")
    
    # State
    throttle = 0.0
    gimbal_pitch = 0.0
    gimbal_roll = 0.0
    rcs_yaw = 0.0
    legs_deployed = -1.0 # -1 retracted, 1 deployed
    
    while True:
        keys = p.getKeyboardEvents()
        
        # Throttle Logic (Incremental)
        if ord('w') in keys and keys[ord('w')] & p.KEY_IS_DOWN:
            throttle += 0.02 # Faster response
        if ord('s') in keys and keys[ord('s')] & p.KEY_IS_DOWN:
            throttle -= 0.02
        throttle = np.clip(throttle, 0.0, 1.0)
        
        # Gimbal Logic (Direct Control)
        # Pitch
        if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
            gimbal_pitch = -1.0 # Forward?
        elif p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
            gimbal_pitch = 1.0 # Back?
        else:
            gimbal_pitch = 0.0
            
        # Roll
        if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
            gimbal_roll = -1.0
        elif p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
            gimbal_roll = 1.0
        else:
            gimbal_roll = 0.0
            
        # RCS Yaw
        if ord('q') in keys and keys[ord('q')] & p.KEY_IS_DOWN:
            rcs_yaw = -1.0
        elif ord('e') in keys and keys[ord('e')] & p.KEY_IS_DOWN:
            rcs_yaw = 1.0
        else:
            rcs_yaw = 0.0
            
        # Legs
        if ord('g') in keys and keys[ord('g')] & p.KEY_WAS_TRIGGERED:
            legs_deployed *= -1.0
            print(f"Legs: {'Deployed' if legs_deployed > 0 else 'Retracted'}")
            
        # Reset
        if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
            env.reset()
            throttle = 0.0
            print("Reset!")
            continue

        # Action Map
        # 0: Throttle [-1, 1] -> Mapped to 0..1 in env, so inputs -1..1
        # Env logic: throttle = (action[0] + 1) / 2.
        # So action[0] = throttle * 2 - 1
        
        action_throttle = throttle * 2.0 - 1.0
        
        # Action: [Throt, GPitch, GRoll, RPitch, RYaw, RRoll, Legs]
        action = np.array([
            action_throttle, 
            gimbal_pitch, 
            gimbal_roll, 
            0.0, # RCS Pitch
            rcs_yaw, 
            0.0, # RCS Roll
            legs_deployed
        ], dtype=np.float32)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            print(f"Episode Done. Reward: {reward}")
            # Do not quit, just reset
            time.sleep(1.0)
            env.reset()
            throttle = 0.0

        time.sleep(1./60.)

if __name__ == "__main__":
    main()
