import gymnasium as gym
import pybullet as p
import time
import numpy as np
import rocket_lander # Register env

def test_physics():
    # Render mode human to see the explosion if any
    env = gym.make("RocketLander-v0", render_mode="human")
    obs, _ = env.reset()
    
    print("🌟 Physics Diagnostic Test Started")
    print("Checking for spontaneous torque/acceleration...")
    
    # disable control
    action = np.zeros(7)
    
    rocketId = env.unwrapped.rocketId
    
    for i in range(240): # 4 seconds
        # Apply zero action
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Get Telemetry
        pos, orn = p.getBasePositionAndOrientation(rocketId)
        lin_vel, ang_vel = p.getBaseVelocity(rocketId)
        
        # Check Contact Points
        contacts = p.getContactPoints(rocketId)
        if (i % 60 == 0) or (len(contacts) > 0 and i < 5):
            print(f"\nStep {i}:")
            print(f"  Pos: {pos}")
            print(f"  AngVel: {ang_vel}")
            if len(contacts) > 0:
                print(f"  🚨 CONTACT DETECTED! Count: {len(contacts)}")
                for c in contacts:
                    # linkIndexA, linkIndexB
                    print(f"    BodyA:{c[1]} LinkA:{c[3]} | BodyB:{c[2]} LinkB:{c[4]} | Dist:{c[8]:.4f}")
            else:
                print("  No Contacts.")
                
        # Check Kinetic Energy/Velocity Spike
        if np.linalg.norm(ang_vel) > 1.0:
            print(f"  🚨 VALIDATION FAILED: High Angular Velocity Detected! {ang_vel}")
            break
            
        time.sleep(1/240.0)
        
    env.close()

if __name__ == "__main__":
    test_physics()
