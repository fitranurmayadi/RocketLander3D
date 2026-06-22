import gymnasium as gym
import rocket_lander
import pybullet as p
import numpy as np
import time
import sys

def verify_full_system():
    print("\n========================================")
    print("   ROCKET DYNAMICS VERIFICATION (FINAL) ")
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
    # 1. GRAVITY CHECK
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
    print(f"  Observed Accel Z: {accel_z:.2f} m/s^2 (Expect > 5.0)")
    check(accel_z > 5.0, 5.0, "Thrust Overcomes Gravity")

    # ==========================
    # 3. RCS PITCH (Local Y Axis)
    # ==========================
    # Nose Up (+Pitch) -> Rot +Y. Action: [4],[7] = 1.0
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
    # 4. RCS YAW (Local Z Axis)
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
    # 5. RCS ROLL (Local X Axis)
    # ==========================
    # Roll via X Axis Torque
    act_roll_p = neutral.copy(); act_roll_p[5] = 1.0; act_roll_p[10] = 1.0
    res = eval_action("RCS Roll Test (+) -> Check Dominant Axis", act_roll_p)
    
    print(f"  AngVel [X, Y, Z]: {res['end_ang']}")
    if abs(res['end_ang'][0]) > abs(res['end_ang'][1]) and abs(res['end_ang'][0]) > abs(res['end_ang'][2]):
        print("  -> Dominant Axis: X (Correct)")
        check(True, True, "Roll actuates X-Axis")
    else:
        print("  -> FAIL: Roll action did not dominate X axis.")

    # ==========================
    # 6. GIMBAL PITCH CHECK
    # ==========================
    # Gimbal Pitch Controls Body Pitch (Rotation around Y)
    # Action[1]
    # Throttle set to 50% (Action 0.0) for stability check
    
    # Gimbal Pitch + (Nozzle Deflects)
    act_gimbal_p_pos = neutral.copy(); act_gimbal_p_pos[0]=0.0; act_gimbal_p_pos[1] = 1.0 
    res = eval_action("Gimbal Pitch (+) [Throttle=50%]", act_gimbal_p_pos)
    
    ang_y = res["end_ang"][1]
    print(f"  AngVel Y: {ang_y:.3f} rad/s")
    check(abs(ang_y) > 0.05, 0.05, "Gimbal Pitch Actuates Y-Axis")
    if ang_y > 0: print("  -> Direction: +Y (Nose Up)")
    else: print("  -> Direction: -Y (Nose Down)")

    # Gimbal Pitch -
    act_gimbal_p_neg = neutral.copy(); act_gimbal_p_neg[0]=0.0; act_gimbal_p_neg[1] = -1.0
    res = eval_action("Gimbal Pitch (-) [Throttle=50%]", act_gimbal_p_neg)
    
    ang_y_neg = res["end_ang"][1]
    print(f"  AngVel Y: {ang_y_neg:.3f} rad/s")
    check(abs(ang_y_neg) > 0.05, 0.05, "Gimbal Pitch (-) Actuates Y-Axis")
    
    # Verify Opposition
    if np.sign(ang_y) != np.sign(ang_y_neg):
        print("  -> PASS: +/- Directions are opposite.")
    else:
        print("  -> FAIL: +/- Directions are SAME (Check Joint Limits/Type).")

    # ==========================
    # 7. GIMBAL ROLL CHECK (Body Roll)
    # ==========================
    # Gimbal Roll (Action[2]) -> Rotates Nozzle Left/Right -> Torque around X -> Body Roll
    
    act_gimbal_r_pos = neutral.copy(); act_gimbal_r_pos[0]=0.0; act_gimbal_r_pos[2] = 1.0
    res = eval_action("Gimbal Roll (+) [Throttle=50%]", act_gimbal_r_pos)
    
    ang_x = res["end_ang"][0]
    print(f"  AngVel X (Roll): {ang_x:.3f} rad/s")
    
    check(abs(ang_x) > 0.05, 0.05, "Gimbal Roll Actuates X-Axis")
    if ang_x > 0: print("  -> Direction: +X")
    else: print("  -> Direction: -X")
    
    # Gimbal Roll -
    act_gimbal_r_neg = neutral.copy(); act_gimbal_r_neg[0]=0.0; act_gimbal_r_neg[2] = -1.0
    res = eval_action("Gimbal Roll (-) [Throttle=50%]", act_gimbal_r_neg)
    
    ang_x_neg = res["end_ang"][0]
    print(f"  AngVel X (Roll): {ang_x_neg:.3f} rad/s")
    check(abs(ang_x_neg) > 0.05, 0.05, "Gimbal Roll (-) Actuates X-Axis")
    
     # Verify Opposition
    if np.sign(ang_x) != np.sign(ang_x_neg):
        print("  -> PASS: +/- Directions are opposite.")
    else:
        print("  -> FAIL: +/- Directions are SAME.")

    # ==========================
    # 8. LANDING SENSOR CHECK
    # ==========================
    print("\n[TEST] Landing Sensor (Drop Test)")
    sys.stdout.flush()
    
    # Reset low to ground with Legs Deployed
    # Options: initial_pos=[0,0,5.0]
    env.reset(options={"initial_pos": [0,0,5.0], "initial_orn": [0,0,0,1]})
    
    # Action: Gravity only, Legs Down (Action[15]=1.0)
    drop_action = neutral.copy()
    drop_action[15] = 1.0 
    
    contact_detected = False
    feet_names = ["X+", "X-", "Y+", "Y-"]
    
    for i in range(100): # 3 seconds max
        obs, _, terminated, _, _ = env.step(drop_action)
        
        # obs[13:17] are foot contacts
        contacts = obs[13:17]
        
        if np.sum(contacts) > 0:
            contact_detected = True
            print(f"  -> Contact Detected at Step {i} | Alt: {obs[17]:.2f}")
            print(f"  -> Foot Report: {[f'{n}:{c}' for n,c in zip(feet_names, contacts)]}")
            break
            
        if terminated:
            print("  -> Terminated before contact detection.")
            break
            
    check(contact_detected, True, "Landing Sensors Trigger on Impact")

    env.close()
    print("\n----------------------------------------")
    print("VERIFICATION COMPLETE")
    print("----------------------------------------")

if __name__ == "__main__":
    verify_full_system()
