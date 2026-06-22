import gymnasium as gym
import time
import numpy as np
import pybullet as p
import rocket_lander
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

def verify_dynamics_visual():
    logger.info("========================================")
    logger.info("      VISUAL DYNAMICS VERIFICATION      ")
    logger.info("========================================")
    
    # Enable Human Rendering
    env = gym.make("RocketLander-v0", render_mode="human")
    
    def spawn_reset():
        # Spawn high enough to see
        return env.reset(options={"initial_pos": [0,0,100], "initial_orn": p.getQuaternionFromEuler([0,0,0])})

    def get_velocity():
        lin, ang = p.getBaseVelocity(env.unwrapped.rocketId)
        return np.array(lin), np.array(ang)

    def run_test(name, action, duration=2.0):
        logger.info(f"TEST: {name}...")
        spawn_reset()
        
        # Settle
        for _ in range(20): 
            env.step(np.zeros(7))
            time.sleep(1/60)
            
        # Apply Action
        for _ in range(int(duration * 30)):
            env.step(action)
            time.sleep(1/30) # Real-time speed
            
        # Check result
        end_lin, end_ang = get_velocity()
        return end_lin, end_ang

    # ==========================
    # RCS TESTS
    # ==========================
    logger.info("--- RCS TESTS ---")
    
    # 1. ROTATE LEFT (YAW +)
    action = np.zeros(7)
    action[0] = -1.0
    action[4] = 1.0 
    _, v_ang = run_test("RCS -> ROTATE LEFT (Yaw+)", action)
    status = "PASS" if v_ang[2] > 0.05 else "FAIL"
    logger.info(f"Result: {status} (YawRate={v_ang[2]:.2f})\n")
    time.sleep(1)

    # 2. TILT BACK (PITCH +)
    action = np.zeros(7)
    action[0] = -1.0
    action[3] = 1.0 
    _, v_ang = run_test("RCS -> TILT BACK (Pitch+)", action)
    status = "PASS" if v_ang[1] < -0.05 else "FAIL"
    logger.info(f"Result: {status} (PitchRate={v_ang[1]:.2f})\n")
    time.sleep(1)

    # 3. TILT RIGHT (ROLL +)
    action = np.zeros(7)
    action[0] = -1.0
    action[5] = 1.0
    _, v_ang = run_test("RCS -> TILT RIGHT (Roll+)", action)
    status = "PASS" if v_ang[0] > 0.05 else "FAIL"
    logger.info(f"Result: {status} (RollRate={v_ang[0]:.2f})\n")
    time.sleep(1)
    
    # ==========================
    # GIMBAL TESTS
    # ==========================
    logger.info("--- GIMBAL TESTS (Throttle 50%) ---")
    
    # 4. GIMBAL TILT BACK (PITCH +)
    # Pitching Gimbal + (Nozzle Back?) -> Thrust Forward -> Torque Back (Nose Up)
    action = np.zeros(7)
    action[0] = 0.0 # 50% Throttle needed for Gimbal to work!
    action[1] = 1.0 # Gimbal Pitch +
    _, v_ang = run_test("GIMBAL -> TILT BACK (Pitch+)", action)
    
    # Gimbal Pitch usually aligns with Rocket Pitch control IF mapped correctly.
    # Let's see what happens physically. 
    # If Nozzle moves +Y, Thrust has +Y component. Force at tail is +Y.
    # Torque = r x F = (0,0,-2.8) x (0, F_y, F_z).
    # Torque X = -(-2.8)*F_y = +2.8*F_y.
    # Wait, Nozzle moving +Y (Sideways) creates Roll Torque.
    # Nozzle moving +X (Forward) creates Pitch Torque.
    # Let's check joint axis again.
    # Pitch Joint Axis: 0 1 0 (Y axis). Rotating around Y moves Nozzle in X/Z plane.
    # So Pitch Joint -> Force X -> Torque Y (Pitch). Correct.
    # Roll Joint Axis: 1 0 0 (X axis). Rotating around X moves Nozzle in Y/Z plane.
    # So Roll Joint -> Force Y -> Torque X (Roll). Correct.
    
    status = "PASS" if abs(v_ang[1]) > 0.01 else "FAIL (Check Direction visually)"
    logger.info(f"Result: {status} (PitchRate={v_ang[1]:.2f})\n")
    time.sleep(1)
    
    # 5. GIMBAL TILT RIGHT (ROLL +)
    action = np.zeros(7)
    action[0] = 0.0 # 50% Throttle
    action[2] = 1.0 # Gimbal Roll +
    _, v_ang = run_test("GIMBAL -> TILT RIGHT/LEFT (Roll+)", action)
    
    status = "PASS" if abs(v_ang[0]) > 0.01 else "FAIL (Check Direction visually)"
    logger.info(f"Result: {status} (RollRate={v_ang[0]:.2f})\n")
    time.sleep(1)

    logger.info("Verification Complete.")
    env.close()

if __name__ == "__main__":
    verify_dynamics_visual()
