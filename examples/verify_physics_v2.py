import gymnasium as gym
import rocket_lander
import pybullet as p
import numpy as np
import time
import sys

def verify_physics_v2():
    print("\n========================================")
    print("   ROCKET DYNAMICS VERIFICATION V2      ")
    print("========================================")
    sys.stdout.flush()
    
    # Use Direct mode for stability
    env = gym.make("RocketLander-v0")
    
    def reset_and_stabilize(height=100.0):
        # Reset high up
        env.reset(options={"initial_pos": [0,0,height], "initial_orn": [0,0,0,1]})
        # Stabilize for a few steps
        neutral = np.full(16, -1.0)
        neutral[1] = 0.0 # Gimbal P Center
        neutral[2] = 0.0 # Gimbal R Center
        for _ in range(10): 
            env.step(neutral)
        return neutral

    def eval_action(name, action, duration=1.0):
        print(f"\n[TEST] {name}")
        sys.stdout.flush()
        
        reset_and_stabilize()
        
        # Capture Start State
        start_lin, start_ang = p.getBaseVelocity(env.unwrapped.rocketId)
        start_pos, _ = p.getBasePositionAndOrientation(env.unwrapped.rocketId)
        
        # Run Action
        steps = int(duration * 30)
        for _ in range(steps):
            env.step(action)
            
        # Capture End State
        end_lin, end_ang = p.getBaseVelocity(env.unwrapped.rocketId)
        end_pos, _ = p.getBasePositionAndOrientation(env.unwrapped.rocketId)
        
        return {
            "start_lin": np.array(start_lin), 
            "end_lin": np.array(end_lin),
            "start_ang": np.array(start_ang),
            "end_ang": np.array(end_ang),
            "start_pos": np.array(start_pos),
            "end_pos": np.array(end_pos)
        }

    def check(val, threshold, name):
        status = "PASS" if val else "FAIL"
        print(f"  -> {name}: {status}")
        return val

    # ==========================
    # 1. GRAVITY V CHECK
    # ==========================
    neutral = np.full(16, -1.0); neutral[1]=0; neutral[2]=0
    res = eval_action("Gravity (Freefall)", neutral, duration=1.5)
    
    accel_z = (res["end_lin"][2] - res["start_lin"][2]) / 1.5
    print(f"  Observed Accel Z: {accel_z:.2f} m/s^2")
    check(accel_z < -9.0, -9.0, "Gravity Pulls Down")

    # ==========================
    # 2. MAIN ENGINE THRUST
    # ==========================
    thrust_act = neutral.copy(); thrust_act[0] = 1.0 # Full Throttle
    res = eval_action("Main Engine (Full Throttle)", thrust_act, duration=1.5)
    
    accel_z = (res["end_lin"][2] - res["start_lin"][2]) / 1.5
    # Force = Mass * Accel -> 60000N / 3000kg = 20m/s^2. Net = 20 - 9.8 = 10.2 m/s^2
    print(f"  Observed Accel Z: {accel_z:.2f} m/s^2 (Expect > 5.0)")
    check(accel_z > 5.0, 5.0, "Thrust Overcomes Gravity")

    # ==========================
    # 3. RCS PITCH (Local Y Axis)
    # ==========================
    # Nose Up (+Pitch) -> Rot +Y. Action: [4],[7] = 1.0
    # Ref: rocket_lander_env.py mapping
    act_pitch_up = neutral.copy(); act_pitch_up[4] = 1.0; act_pitch_up[7] = 1.0
    res = eval_action("RCS Pitch Up (+) -> Expect AngVel Y > 0", act_pitch_up)
    
    ang_y = res["end_ang"][1]
    print(f"  AngVel Y: {ang_y:.3f} rad/s")
    check(ang_y > 0.05, 0.05, "Pitch Positive Response")

    # Nose Down (-Pitch) -> Rot -Y. Action: [3],[8] = 1.0
    act_pitch_down = neutral.copy(); act_pitch_down[3] = 1.0; act_pitch_down[8] = 1.0
    res = eval_action("RCS Pitch Down (-) -> Expect AngVel Y < 0", act_pitch_down)
    
    ang_y = res["end_ang"][1]
    print(f"  AngVel Y: {ang_y:.3f} rad/s")
    check(ang_y < -0.05, -0.05, "Pitch Negative Response")

    # ==========================
    # 4. RCS YAW (Local Z Axis? wait, Rocket Z is vertical. Yaw is rotation around Z)
    # ==========================
    # Yaw Left (+) -> Rot +Z. Action: [11],[12] = 1.0
    act_yaw_left = neutral.copy(); act_yaw_left[11] = 1.0; act_yaw_left[12] = 1.0
    res = eval_action("RCS Yaw Left (CCW) (+) -> Expect AngVel Z > 0", act_yaw_left)
    
    ang_z = res["end_ang"][2]
    print(f"  AngVel Z: {ang_z:.3f} rad/s")
    check(ang_z > 0.05, 0.05, "Yaw Positive Response")

    # Yaw Right (-) -> Rot -Z. Action: [13],[14] = 1.0
    act_yaw_right = neutral.copy(); act_yaw_right[13] = 1.0; act_yaw_right[14] = 1.0
    res = eval_action("RCS Yaw Right (CW) (-) -> Expect AngVel Z < 0", act_yaw_right)
    
    ang_z = res["end_ang"][2]
    print(f"  AngVel Z: {ang_z:.3f} rad/s")
    check(ang_z < -0.05, -0.05, "Yaw Negative Response")

    # ==========================
    # 5. RCS ROLL (Local X Axis? wait, typical Rocket Frame is Z-up)
    # ==========================
    # In documentation: "Roll: Spinning around long vertical axis".
    # Wait. Docs say:
    # "Roll (Google: Z-Axis): Spinning around long vertical axis."
    # "Pitch (Global Y-Axis): Nose tips Forward/Backward."
    # "Yaw (Global X-Axis): Nose tips Left/Right."
    
    # BUT PyBullet Body Frame:
    # Cylinder is usually aligned with Z?
    # Let's check AngVel indices.
    # If rocket is vertical (0,0,1 orientation), then:
    # Rotation around Z (Vertical) IS Roll/Yaw?
    # Usually:
    # Roll = Rotation around longitudinal axis (Z).
    # Pitch = Rotation around lateral axis (Y).
    # Yaw = Rotation around normal axis (X).

    # Let's trust the "Spin" vs "Tilt" behavior.
    
    # Test: Action "Roll" [5],[10] (Right Wing Down / Clockwise?).
    # If this creates rotation around X, then "Roll" in code = Rotation X.
    act_roll_p = neutral.copy(); act_roll_p[5] = 1.0; act_roll_p[10] = 1.0
    res = eval_action("RCS Roll Test (+) -> Check Axis", act_roll_p)
    
    print(f"  AngVel [X, Y, Z]: {res['end_ang']}")
    if abs(res['end_ang'][0]) > abs(res['end_ang'][1]) and abs(res['end_ang'][0]) > abs(res['end_ang'][2]):
        print("  -> Dominant Axis: X (Correct for 'Roll' if X is one of lateral axes)")
        check(True, True, "Roll actuates X-Axis")
        if res['end_ang'][0] > 0: print("  -> Direction: +X")
        else: print("  -> Direction: -X")
    else:
        print("  -> FAIL: Roll action did not dominate X axis.")

    env.close()
    print("\n----------------------------------------")
    print("VERIFICATION COMPLETE")
    print("----------------------------------------")

if __name__ == "__main__":
    verify_physics_v2()
