import gymnasium as gym
import numpy as np
import pybullet as p
import time
import os
import sys
# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rocket_lander

class PID:
    def __init__(self, kp, ki, kd, setpoint=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.prev_error = 0.0
        self.integral = 0.0
    
    def update(self, measurement, dt):
        error = self.setpoint - measurement
        # Anti-windup sederhana untuk integral
        self.integral += error * dt
        self.integral = np.clip(self.integral, -5.0, 5.0) 
        
        derivative = (error - self.prev_error) / dt
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        self.prev_error = error
        return output

def run_simulation(env, pid_params, mode='pitch_rcs', visualize=False):
    """
    Menjalankan 1 episode simulasi untuk mengetes PID.
    """
    # --- 1. SETUP KONDISI AWAL (INITIAL STATE) ---
    initial_pos = [0, 0, 100] 
    initial_rpy = [0, 0, 0]   
    target_val = 0.0

    if 'pitch' in mode:
        initial_rpy = [0, np.radians(45), 0] # Sudut gangguan diperkecil biar aman
    elif 'roll' in mode:
        initial_rpy = [np.radians(45), 0, 0] 
    elif 'yaw' in mode:
        initial_rpy = [0, 0, np.radians(30)] 
    elif 'altitude' in mode:
        initial_pos = [0, 0, 10] 
        target_val = 30.0        
    
    initial_quat = p.getQuaternionFromEuler(initial_rpy)
    obs, _ = env.reset(options={"initial_pos": initial_pos, "initial_orn": initial_quat})
    
    pid = PID(pid_params[0], pid_params[1], pid_params[2], setpoint=target_val)
    
    total_error = 0.0
    # max_steps dikurangi sedikit agar proses tuning lebih cepat
    max_steps = 300 
    
    dt = env.unwrapped.dt * env.unwrapped.STEPS_PER_CONTROL

    for _ in range(max_steps):
        if visualize:
            env.render()
            time.sleep(1/120)

        # --- 2. SENSOR READING ---
        pos = obs[0:3]
        quat = obs[3:7]
        roll, pitch, yaw = p.getEulerFromQuaternion(quat)
        
        action = np.zeros(16)
        action[0] = -1.0 
        
        measurement = 0.0
        control_signal = 0.0

        # --- 3. CONTROL LOGIC (LOGIKA DIPERBAIKI DISINI) ---
        
        # ========== PITCH CONTROL ==========
        if 'pitch' in mode:
            measurement = pitch
            control_signal = pid.update(measurement, dt)
            control_signal = np.clip(control_signal, -1.0, 1.0)
            
            # Action Mixing
            if 'rcs' in mode or 'combined' in mode:
                # Jika PID (+) -> Kita butuh Pitch Up -> Nyalakan Thruster Atas (4 & 7)
                if control_signal > 0:
                     action[4] = abs(control_signal); action[7] = abs(control_signal)
                else:
                     action[3] = abs(control_signal); action[8] = abs(control_signal)
            
            if 'gimbal' in mode or 'combined' in mode:
                action[0] = 0.0 
                # Positive Gimbal Pitch = Negative Moment (Nose Down)
                # If error is positive (Nose Down needed), control_signal is positive.
                # We want positive action[1].
                action[1] = control_signal

        # ========== ROLL CONTROL ==========
        elif 'roll' in mode:
            measurement = roll
            control_signal = pid.update(measurement, dt)
            control_signal = np.clip(control_signal, -1.0, 1.0)

            # Action Mixing
            if 'rcs' in mode or 'combined' in mode:
                # Jika PID (+) -> Kita butuh Roll Right -> Nyalakan Thruster (6 & 9)
                if control_signal > 0:
                    action[6] = abs(control_signal); action[9] = abs(control_signal)
                else:
                    action[5] = abs(control_signal); action[10] = abs(control_signal)

            if 'gimbal' in mode or 'combined' in mode:
                action[0] = 0.0 
                # Positive Gimbal Roll = Positive Moment (Roll Right)
                # If error is positive (Roll Right needed), control_signal is positive.
                # We want positive action[2].
                action[2] = control_signal

        # ========== YAW CONTROL ==========
        elif 'yaw' in mode:
            measurement = yaw
            control_signal = pid.update(measurement, dt)
            control_signal = np.clip(control_signal, -1.0, 1.0)
            
            if control_signal > 0:
                action[13] = abs(control_signal); action[14] = abs(control_signal) # Yaw - (Right)
            else:
                action[11] = abs(control_signal); action[12] = abs(control_signal) # Yaw + (Left)

        # ========== ALTITUDE CONTROL ==========
        elif 'altitude' in mode:
            measurement = pos[2] 
            control_signal = pid.update(measurement, dt)
            
            base_throttle = -0.55 
            throttle_cmd = base_throttle + control_signal
            action[0] = np.clip(throttle_cmd, -1.0, 1.0)
            
            p.resetBasePositionAndOrientation(env.unwrapped.rocketId, pos, [0,0,0,1])

        # --- 4. STEP ENVIRONMENT ---
        obs, reward, terminated, truncated, _ = env.step(action)
        
        # --- 5. COST CALCULATION ---
        # Penalize error AND use of control (fuel/effort)
        total_error += (target_val - measurement)**2 + 0.05 * (control_signal**2)
        
        # Penalti jika tumbling terlalu parah (keamanan)
        if abs(roll) > 1.5 or abs(pitch) > 1.5:
            total_error += 10000
            break

        if terminated: 
            total_error += 5000 
            break
            
    return total_error

def twiddle(env, mode, initial_params=[1.0, 0.0, 1.0], tol=0.01):
    """ Algoritma Optimasi Coordinate Descent """
    p_params = list(initial_params)
    dp = [0.01, 0.01, 0.01] # Step pencarian awal [Kp, Ki, Kd]
    
    best_err = run_simulation(env, p_params, mode)
    print(f"--> Tuning {mode.upper()}... Start Error: {best_err:.2f}")
    
    it = 0
    while sum(dp) > tol:
        for i in range(3):
            p_params[i] += dp[i]
            err = run_simulation(env, p_params, mode)
            
            if err < best_err:
                best_err = err
                dp[i] *= 1.1
            else:
                p_params[i] -= 2 * dp[i]
                err = run_simulation(env, p_params, mode)
                
                if err < best_err:
                    best_err = err
                    dp[i] *= 1.1
                else:
                    p_params[i] += dp[i]
                    dp[i] *= 0.9
        it += 1
            
    return p_params
if __name__ == "__main__":
    env = gym.make("RocketLander-v0", render_mode="rgb_array")
    
    print("============================================")
    print("   FULL SYSTEM PID AUTO-TUNER (FIXED)       ")
    print("============================================")
    
    # KITA RESET PARAMETER KE NILAI "SAFE START"
    # Agar tuning dimulai dari kondisi stabil
    safe_params = [1.5, 0.0, 1.0] 

    # --- 1. ROLL TUNING ---
    print("\n--- PHASE 1: ROLL CONTROL ---")
    roll_rcs = twiddle(env, 'roll_rcs', safe_params)
    print(f" [RESULT] Roll RCS:      {roll_rcs}")
    
    roll_gim = twiddle(env, 'roll_gimbal', safe_params)
    print(f" [RESULT] Roll Gimbal:   {roll_gim}")
    
    roll_comb = twiddle(env, 'roll_combined', roll_gim) # Start from gimbal result
    print(f" [RESULT] Roll Combined: {roll_comb}")

    # --- 2. PITCH TUNING ---
    print("\n--- PHASE 2: PITCH CONTROL ---")
    pitch_rcs = twiddle(env, 'pitch_rcs', safe_params)
    print(f" [RESULT] Pitch RCS:     {pitch_rcs}")
    
    pitch_gim = twiddle(env, 'pitch_gimbal', safe_params)
    print(f" [RESULT] Pitch Gimbal:  {pitch_gim}")
    
    pitch_comb = twiddle(env, 'pitch_combined', pitch_gim) # Start from gimbal result
    print(f" [RESULT] Pitch Combined: {pitch_comb}")

    # --- 3. YAW TUNING ---
    print("\n--- PHASE 3: YAW CONTROL ---")
    yaw_rcs = twiddle(env, 'yaw_rcs', [1.0, 0.0, 0.5])
    print(f" [RESULT] Yaw RCS:       {yaw_rcs}")

    # --- 4. ALTITUDE TUNING ---
    print("\n--- PHASE 4: ALTITUDE CONTROL ---")
    alt_pid = twiddle(env, 'altitude', [0.8, 0.1, 1.5])
    print(f" [RESULT] Altitude:      {alt_pid}")

    env.close()

    # --- VISUALISASI ---
     # --- 5. VISUALISASI HASIL ---

    print("\n============================================")

    print(" VISUALIZING TUNED CONTROLLERS... ")

    print("============================================")

    env_gui = gym.make("RocketLander-v0", render_mode="human")

    # Gunakan variabel hasil di atas (yang sudah di-update twiddle)

    # Jika ingin langsung pakai nilai hardcoded dari log tanpa twiddle lagi,

    # ganti variabel di bawah dengan list angka manual.

    # Test Roll (Combined)

    print("Visualizing: Roll Stabilization (Combined)")

    run_simulation(env_gui, roll_comb, 'roll_combined', visualize=True)

    # Test Pitch (Combined)

    print("Visualizing: Pitch Stabilization (Combined)")

    run_simulation(env_gui, pitch_comb, 'pitch_combined', visualize=True)

    # Test Yaw

    print("Visualizing: Yaw Stabilization")

    run_simulation(env_gui, yaw_rcs, 'yaw_rcs', visualize=True)


    # Test Altitude

    print("Visualizing: Altitude Hold")

    run_simulation(env_gui, alt_pid, 'altitude', visualize=True)


    env_gui.close() 