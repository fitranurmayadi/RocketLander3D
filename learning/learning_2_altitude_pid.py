"""
Learning Lesson 2: Altitude PID Control & Hovering

This script implements a basic Proportional-Derivative (PD) loop to control 
the vertical altitude of the rocket, enabling it to hover at a target height.

Concepts covered:
1. Proportional Control (Kp): Acceleration proportional to position error.
2. Derivative Control (Kd): Damping vertical speed to prevent overshoot.
3. Gravity Feedforward (gravity_ff): Constant baseline throttle to counteract gravity.
4. Cascaded PID Loops: Using position error to command velocity, then velocity error to command throttle.
"""

import gymnasium as gym
import pybullet as p
import numpy as np
import time

try:
    import rocket_lander
except ImportError:
    pass

class SimplePID:
    """A standard PID controller implementation."""
    def __init__(self, kp, ki, kd, dt, limit=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.limit = limit
        self.integral = 0.0
        self.prev_error = 0.0
        
    def update(self, error) -> float:
        # 1. Proportional term
        p_term = self.kp * error
        
        # 2. Integral term with windup limit (anti-windup)
        self.integral += error * self.dt
        self.integral = max(-5.0, min(5.0, self.integral)) # clamp integral
        i_term = self.ki * self.integral
        
        # 3. Derivative term
        d_term = self.kd * (error - self.prev_error) / self.dt
        self.prev_error = error
        
        # Total control output clamped to limits
        output = p_term + i_term + d_term
        return max(-self.limit, min(self.limit, output))

def main():
    print("==================================================")
    print("Lesson 2: Altitude PID Control & Hovering")
    print("==================================================")
    print("This script will guide the rocket to fly up to Z=50.0m")
    print("and hover there automatically using a cascaded PID loop.")
    print("==================================================")
    
    # Instantiate env. We use v1 (mission mode) or v0.
    # v0 randomize position spawn: let's use v0 but reset at a known position.
    env = gym.make("RocketLander-v0", render_mode="human")
    obs, _ = env.reset()
    
    # Override position to spawn at launchpad Z=3.0m
    rocket_id = env.unwrapped.rocketId
    p.resetBasePositionAndOrientation(rocket_id, [0.0, 0.0, 3.0], [0.0, 0.0, 0.0, 1.0])
    p.resetBaseVelocity(rocket_id, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    obs, _, _, _, _ = env.step(np.zeros(16))
    
    # Target Altitude
    target_alt = 50.0
    
    # 60Hz control loop step time
    dt = 1.0 / 60.0
    
    # Cascaded Loops setup:
    # Outer Loop: Altitude Error -> Desired Vertical Velocity (Vz)
    alt_pid = SimplePID(kp=1.5, ki=0.0, kd=0.0, dt=dt, limit=20.0)
    
    # Inner Loop: Vz Error -> Throttle adjustment
    vz_pid = SimplePID(kp=0.8, ki=0.15, kd=0.0, dt=dt, limit=1.0)
    
    # Gravity feedforward (ff) is about 9.81% throttle because the thrust-to-weight ratio (TWR)
    # is 10.19 at full thrust. (Hover thrust = Weight => Throttle = 1 / TWR = 1 / 10.19 = 9.81%)
    gravity_ff = 0.0981
    
    print(f"Gravity Feedforward: {gravity_ff*100:.2f}% throttle")
    
    try:
        for step in range(1200): # Run for 20 seconds (1200 steps at 60Hz)
            # 1. Parse current state
            z_curr = obs[2]
            vz_curr = obs[9]
            
            # 2. Outer Loop: Calculate desired vz to correct altitude error
            alt_error = target_alt - z_curr
            desired_vz = alt_pid.update(alt_error)
            
            # 3. Inner Loop: Calculate throttle adjustment to track desired velocity
            vz_error = desired_vz - vz_curr
            throttle_adj = vz_pid.update(vz_error)
            
            # 4. Total throttle = gravity compensation + PID correction
            throttle = gravity_ff + throttle_adj
            throttle = max(0.0, min(1.0, throttle))
            
            # 5. Build Gymnasium action
            action = np.zeros(16)
            action[0] = throttle * 2.0 - 1.0 # map [0, 1] throttle to [-1, 1] action
            
            # Keep orientation locked upright to focus on altitude only
            # Simple attitude locks (cheating for this lesson using helper torque force):
            p.applyExternalTorque(rocket_id, -1, [-500.0*obs[3], -500.0*obs[4], -500.0*obs[12]], p.LINK_FRAME)
            
            # Step the simulation
            obs, reward, terminated, truncated, info = env.step(action)
            
            if step % 10 == 0:
                print(f"Time: {step*dt:.1f}s | Alt: {z_curr:.2f}m | Error: {alt_error:.2f}m | Vz: {vz_curr:.2f}m/s | Throttle: {throttle*100:.1f}%")
                
            time.sleep(dt)
            
    except KeyboardInterrupt:
        print("\nSimulation ended by user.")
    finally:
        env.close()

if __name__ == "__main__":
    main()
