import gymnasium as gym
import rocket_lander
import pybullet as p
import numpy as np
import time

def check_twr():
    print("--- Verifying TWR (Thrust-to-Weight Ratio) ---")
    
    env = gym.make("RocketLander-v0")
    env.reset()
    
    rocket_id = env.unwrapped.rocketId
    
    # 1. Full Tank TWR
    mass_full = env.unwrapped.MAX_MASS
    weight_full = mass_full * 9.81
    max_thrust = env.unwrapped.MAIN_ENGINE_POWER
    
    twr_full = max_thrust / weight_full
    print(f"\n[Full Tank]")
    print(f"  Mass: {mass_full} kg")
    print(f"  Weight: {weight_full:.2f} N")
    print(f"  Max Thrust: {max_thrust} N")
    print(f"  TWR: {twr_full:.2f} (Expect ~2.0)")
    
    # 2. Empty Tank TWR
    mass_empty = env.unwrapped.MIN_MASS
    weight_empty = mass_empty * 9.81
    
    twr_empty = max_thrust / weight_empty
    print(f"\n[Empty Tank]")
    print(f"  Mass: {mass_empty} kg")
    print(f"  Weight: {weight_empty:.2f} N")
    print(f"  TWR: {twr_empty:.2f} (Expect ~20.0)")

    # 3. Hover Test (Simulation)
    print("\n--- Hover Test (Full Tank) ---")
    # To hover, Thrust = Weight
    # Throttle * MaxThrust = Weight
    # Throttle = Weight / MaxThrust = 1 / TWR
    required_throttle = 1.0 / twr_full
    # Action range -1..1 maps to 0..1
    # 0 -> 0.5. We need (action + 1) / 2 = required_throttle
    # action + 1 = 2 * required_throttle
    # action = 2 * required_throttle - 1
    hover_action_val = 2.0 * required_throttle - 1.0
    
    print(f"Required Throttle %: {required_throttle*100:.2f}%")
    print(f"Action Value: {hover_action_val:.4f}")
    
    env.reset()
    # Force vertical orientation
    p.resetBasePositionAndOrientation(rocket_id, [0, 0, 50], [0, 0, 0, 1])
    p.resetBaseVelocity(rocket_id, [0,0,0], [0,0,0])
    
    action = np.ones(16) * -1.0 # All off (RCS off)
    action[0] = hover_action_val # Set hover throttle
    action[15] = 1.0 # Legs deployed (optional)
    
    # Run for 1 second
    z_start = 50.0
    for _ in range(30):
        env.step(action)
        
    pos, _ = p.getBasePositionAndOrientation(rocket_id)
    z_end = pos[2]
    
    print(f"Start Z: {z_start:.2f}")
    print(f"End Z:   {z_end:.2f}")
    
    drift = z_end - z_start
    print(f"Drift: {drift:.4f} m")
    
    if abs(drift) < 1.0:
        print("PASS: Rocket hovers near stationary.")
    else:
        print("FAIL: Significant vertical drift.")

    env.close()

if __name__ == "__main__":
    check_twr()
