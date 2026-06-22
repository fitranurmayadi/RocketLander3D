import gymnasium as gym
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import os
import sys
# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rocket_lander

# --- CONFIG ---
# Masukkan parameter 'Combined' hasil tuning terakhir Anda di sini
ROLL_PARAMS  = [2.43, 0.17, 4.30] # Kp, Ki, Kd
PITCH_PARAMS = [4.14, 2.01, 4.64] # Kp, Ki, Kd

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
        self.integral += error * dt
        # Clamp integral agar tidak windup parah saat debug
        self.integral = np.clip(self.integral, -2.0, 2.0) 
        derivative = (error - self.prev_error) / dt
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        self.prev_error = error
        return output, (self.kp*error, self.ki*self.integral, self.kd*derivative)

def debug_simulation(env, mode='pitch_combined'):
    # Setup
    initial_rpy = [0, 0, 0]
    pid = None
    
    if mode == 'pitch_combined':
        initial_rpy = [0, np.radians(30), 0] # Miring 30 derajat
        pid = PID(*PITCH_PARAMS)
    elif mode == 'roll_combined':
        initial_rpy = [np.radians(30), 0, 0]
        pid = PID(*ROLL_PARAMS)

    initial_quat = p.getQuaternionFromEuler(initial_rpy)
    obs, _ = env.reset(options={"initial_pos": [0, 0, 50], "initial_orn": initial_quat})
    
    # Data logging
    history = {
        'time': [],
        'angle': [],
        'setpoint': [],
        'pid_out': [],
        'p_term': [], 'i_term': [], 'd_term': [],
        'gimbal_act': [],
        'rcs_act': []
    }
    
    dt = env.unwrapped.dt * env.unwrapped.STEPS_PER_CONTROL
    max_steps = 300 # 2.5 detik

    print(f"\n--- DEBUGGING MODE: {mode.upper()} ---")
    print(f"{'Step':<5} | {'Angle':<8} | {'PID_Out':<8} | {'Gimbal':<8} | {'RCS':<8}")

    for step in range(max_steps):
        # 1. Sensing
        quat = obs[3:7]
        roll, pitch, yaw = p.getEulerFromQuaternion(quat)
        
        measurement = pitch if mode == 'pitch_combined' else roll
        
        # 2. PID Calc
        pid_out, terms = pid.update(measurement, dt)
        pid_out = np.clip(pid_out, -1.0, 1.0)
        
        # 3. Action Logic (Copy dari script tuning)
        action = np.zeros(16)
        action[0] = 0.0 # Throttle 50%
        
        gimbal_val = 0.0
        rcs_val = 0.0
        
        if mode == 'pitch_combined':
            # Gimbal
            action[1] = pid_out
            gimbal_val = action[1]
            
            # RCS logic
            if pid_out > 0:
                action[4] = abs(pid_out); action[7] = abs(pid_out)
                rcs_val = 1.0 # Indikator RCS Positif firing
            else:
                action[3] = abs(pid_out); action[8] = abs(pid_out)
                rcs_val = -1.0 # Indikator RCS Negatif firing
                
        elif mode == 'roll_combined':
            # Gimbal
            action[2] = pid_out
            gimbal_val = action[2]
            
            # RCS Logic
            if pid_out > 0:
                action[5] = abs(pid_out); action[10] = abs(pid_out)
                rcs_val = 1.0
            else:
                action[6] = abs(pid_out); action[9] = abs(pid_out)
                rcs_val = -1.0

        # 4. Step
        obs, _, terminated, _, _ = env.step(action)
        
        # 5. Log
        history['time'].append(step * dt)
        history['angle'].append(np.degrees(measurement))
        history['setpoint'].append(0.0)
        history['pid_out'].append(pid_out)
        history['p_term'].append(terms[0])
        history['i_term'].append(terms[1])
        history['d_term'].append(terms[2])
        history['gimbal_act'].append(gimbal_val)
        history['rcs_act'].append(rcs_val)
        
        if step % 10 == 0:
             print(f"{step:<5} | {np.degrees(measurement):<8.2f} | {pid_out:<8.2f} | {gimbal_val:<8.2f} | {rcs_val:<8.1f}")
             
        if terminated: break

    return history

def plot_results(hist, title):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Plot 1: Response Angle
    ax1.set_title(f"{title} - System Response")
    ax1.plot(hist['time'], hist['angle'], 'b-', label='Angle (Deg)', linewidth=2)
    ax1.plot(hist['time'], hist['setpoint'], 'r--', label='Target')
    ax1.set_ylabel("Angle (deg)")
    ax1.grid(True)
    ax1.legend()
    
    # Plot 2: PID Terms contribution
    ax2.set_title("Internal PID Components")
    ax2.plot(hist['time'], hist['p_term'], label='P Term', alpha=0.5)
    ax2.plot(hist['time'], hist['i_term'], label='I Term', color='green', linewidth=2)
    ax2.plot(hist['time'], hist['d_term'], label='D Term', color='red', alpha=0.5)
    ax2.set_ylabel("Force Contribution")
    ax2.grid(True)
    ax2.legend()
    
    # Plot 3: Actuator Agreement
    ax3.set_title("Actuator Output (Check Polarity!)")
    ax3.plot(hist['time'], hist['gimbal_act'], 'k-', label='Gimbal Cmd')
    ax3.plot(hist['time'], hist['rcs_act'], 'm--', label='RCS Direction (+/-)')
    ax3.set_ylabel("Command (-1 to 1)")
    ax3.set_xlabel("Time (s)")
    ax3.grid(True)
    ax3.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    env = gym.make("RocketLander-v0", render_mode="rgb_array") # Gunakan GUI jika ingin lihat
    
    # Debug Pitch Combined
    hist_pitch = debug_simulation(env, mode='pitch_combined')
    plot_results(hist_pitch, "PITCH Combined Response")
    
    env.close()