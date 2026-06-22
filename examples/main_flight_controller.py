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
    def __init__(self, kp, ki, kd, setpoint=0.0, output_limits=(-1.0, 1.0), integral_limits=(-5.0, 5.0)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.prev_error = 0.0
        self.integral = 0.0
        self.output_min, self.output_max = output_limits
        self.int_min, self.int_max = integral_limits
    
    def update(self, measurement, dt):
        error = self.setpoint - measurement
        
        # Integral term dengan anti-windup clamping
        self.integral += error * dt
        self.integral = np.clip(self.integral, self.int_min, self.int_max)
        
        derivative = (error - self.prev_error) / dt
        
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        
        self.prev_error = error
        return np.clip(output, self.output_min, self.output_max)

class FlightComputer:
    def __init__(self):
        # ==========================================================
        # MASUKKAN HASIL TUNING ANDA DI SINI
        # ==========================================================
        
        # 1. ROLL (Menggunakan nilai 'Combined' agar paling stabil)
        # [RESULT] Roll Combined: Kp=2.77, Ki=-0.05, Kd=4.00
        self.pid_roll = PID(kp=2.77, ki=-0.05, kd=4.00)
        
        # 2. PITCH (Menggunakan nilai 'Combined')
        # [RESULT] Pitch Combined: Kp=3.78, Ki=2.45, Kd=5.12
        self.pid_pitch = PID(kp=3.78, ki=2.45, kd=5.12)
        
        # 3. YAW (RCS Only)
        # [RESULT] Yaw RCS: Kp=1.26, Ki=0.02, Kd=2.35
        self.pid_yaw = PID(kp=1.26, ki=0.02, kd=2.35)
        
        # 4. ALTITUDE (Throttle)
        # [RESULT] Altitude: Kp=0.80, Ki=0.10, Kd=2.00
        self.pid_alt = PID(kp=0.80, ki=0.10, kd=2.00)

        # 5. POSITION HOLD (Bonus: Agar roket mau ke tengah 0,0)
        # Ini belum di-tune otomatis, saya beri nilai konservatif pelan
        self.pid_pos_x = PID(kp=0.1, ki=0.0, kd=0.1, output_limits=(-0.2, 0.2)) # Outputnya target Pitch angle (rad)
        self.pid_pos_y = PID(kp=0.1, ki=0.0, kd=0.1, output_limits=(-0.2, 0.2)) # Outputnya target Roll angle (rad)

    def compute_action(self, obs, dt):
        # Unpack Observation
        # Obs: [Pos(3), Quat(4), Vel(3), AngVel(3), Contacts(4), Alt(1), Fuel(1)]
        pos = obs[0:3]
        quat = obs[3:7]
        roll, pitch, yaw = p.getEulerFromQuaternion(quat)
        
        # --- 1. GUIDANCE LOOP (Posisi -> Target Angle) ---
        # Kita ingin Posisi X,Y = 0. 
        # Untuk ke X+ (Maju), kita perlu Pitch - (Nunduk).
        # Untuk ke Y+ (Kiri), kita perlu Roll + (Miring Kanan) *tergantung frame*.
        
        target_pitch = -self.pid_pos_x.update(pos[0], dt) # Invert logic X -> Pitch
        target_roll  = self.pid_pos_y.update(pos[1], dt)  # Logic Y -> Roll
        
        # Limit target angle agar roket tidak terbalik (max 20 derajat)
        max_tilt = np.radians(20)
        target_pitch = np.clip(target_pitch, -max_tilt, max_tilt)
        target_roll  = np.clip(target_roll, -max_tilt, max_tilt)

        # Update Setpoint Attitude PID
        self.pid_pitch.setpoint = target_pitch
        self.pid_roll.setpoint  = target_roll
        self.pid_yaw.setpoint   = 0.0 # Selalu menghadap depan
        self.pid_alt.setpoint   = 20.0 # Target ketinggian (Hover di 20m dulu)

        # Jika sudah dekat tanah (pos[2] < 30) dan di tengah, turunkan target ketinggian (Landing Mode)
        if pos[2] < 30 and abs(pos[0]) < 5 and abs(pos[1]) < 5:
            self.pid_alt.setpoint = -2.0 # Targetkan sedikit di bawah tanah agar throttle terus menekan sampai touch
        
        # --- 2. CONTROL LOOP (Attitude/Alt -> Actuators) ---
        
        # Altitude Control
        alt_out = self.pid_alt.update(pos[2], dt)
        base_throttle = -0.58 # Feedforward gravity compensator
        throttle_cmd = base_throttle + alt_out
        
        # Attitude Control
        pitch_out = self.pid_pitch.update(pitch, dt)
        roll_out  = self.pid_roll.update(roll, dt)
        yaw_out   = self.pid_yaw.update(yaw, dt)
        
        # --- 3. MIXER (Menggabungkan Output PID ke 16 Action) ---
        action = np.zeros(16)
        
        # A. Main Throttle
        action[0] = np.clip(throttle_cmd, -1.0, 1.0)
        
        # B. TVC (Gimbal) - Menggunakan sebagian output PID Pitch/Roll
        action[1] = np.clip(pitch_out, -1.0, 1.0) # Gimbal Pitch
        action[2] = np.clip(roll_out, -1.0, 1.0)  # Gimbal Roll
        
        # C. RCS (Reaction Control System) - Membantu Gimbal
        # Logika RCS Pitch
        if pitch_out > 0: # PID minta Pitch Up (+)
            # Nyalakan thruster 4 & 7 untuk mendongak (+)
            action[4] = abs(pitch_out); action[7] = abs(pitch_out)
        else:             # PID minta Pitch Down (-)
            # Nyalakan thruster 3 & 8 untuk menunduk (-)
            action[3] = abs(pitch_out); action[8] = abs(pitch_out)
        
        # Logika Gimbal Pitch
        action[1] = np.clip(pitch_out, -1.0, 1.0)
                    
        # Logika Gimbal Roll
        action[2] = np.clip(roll_out, -1.0, 1.0)  
        
        # Logika RCS Roll
        if roll_out > 0: # PID minta Roll Right (+)
            action[6] = abs(roll_out); action[9] = abs(roll_out)
        else:            # PID minta Roll Left (-)
            action[5] = abs(roll_out); action[10] = abs(roll_out)
                    
        # Logika RCS Yaw
        if yaw_out > 0:   # Yaw Left (+)
            action[11] = abs(yaw_out); action[12] = abs(yaw_out)
        else:             # Yaw Right (-)
            action[13] = abs(yaw_out); action[14] = abs(yaw_out)

        # D. Landing Legs
        # Deploy jika ketinggian < 10 meter
        if pos[2] < 10.0:
            action[15] = 1.0
        else:
            action[15] = -1.0
            
        return action

def main():
    # Render Mode Human untuk melihat hasil
    env = gym.make("RocketLander-v0", render_mode="human")
    
    # Reset dengan posisi agak jauh dari pad untuk mengetes Guidance
    # Posisi: X=10m, Y=5m, Z=60m
    initial_pos = [10, 5, 60] 
    obs, _ = env.reset(options={"initial_pos": initial_pos})
    
    computer = FlightComputer()
    
    print("==========================================")
    print("   ROCKET LANDER - AUTONOMOUS LANDING     ")
    print("==========================================")
    print(f"Target: Landing Pad (0,0)")
    
    step_count = 0
    dt = env.unwrapped.dt * env.unwrapped.STEPS_PER_CONTROL

    while True:
        # Hitung aksi dari Flight Computer
        action = computer.compute_action(obs, dt)
        
        # Eksekusi
        obs, reward, terminated, truncated, _ = env.step(action)
        
        # Telemetry Log (setiap 30 frame)
        if step_count % 30 == 0:
            pos = obs[0:3]
            vel = obs[7:10]
            alt = pos[2]
            print(f"Alt: {alt:6.2f}m | VelZ: {vel[2]:5.2f}m/s | PosX: {pos[0]:5.2f}")
            
        step_count += 1
        
        # Reset jika crash atau sukses landing
        if terminated or truncated:
            print("\nSIMULATION ENDED.")
            if obs[13] or obs[14] or obs[15] or obs[16]: # Cek sensor kaki
                print(">>> TOUCHDOWN DETECTED! <<<")
            else:
                print(">>> CRASH / ABORT <<<")
                
            time.sleep(2)
            obs, _ = env.reset(options={"initial_pos": [np.random.uniform(-10,10), np.random.uniform(-10,10), 80]})
            step_count = 0

if __name__ == "__main__":
    main()