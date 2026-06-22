import gymnasium as gym
import time
import numpy as np
import pybullet as p
import rocket_lander
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

def verify_dynamics():
    logger.info("========================================")
    logger.info("      LAPORAN VERIFIKASI DINAMIKA       ")
    logger.info("========================================")
    
    env = gym.make("RocketLander-v0", render_mode=None)
    
    def spawn_reset():
        return env.reset(options={"initial_pos": [0,0,100], "initial_orn": p.getQuaternionFromEuler([0,0,0])})

    def get_velocity():
        lin, ang = p.getBaseVelocity(env.unwrapped.rocketId)
        return np.array(lin), np.array(ang)

    # 1. TEST THRUSTER (NAIK)
    # -----------------------
    spawn_reset()
    for _ in range(30): env.step(np.zeros(7))
    start_v, _ = get_velocity()
    
    action = np.zeros(7)
    action[0] = 1.0 # Throttle Max
    for _ in range(30): env.step(action)
    end_v, _ = get_velocity()
    
    accel_z = end_v[2] - start_v[2]
    status = "OK" if accel_z > 0 else "GAGAL"
    logger.info(f"1. Thruster (Naik):          {status} | Akselerasi = {accel_z:.2f} m/s")

    # 2. TEST PUTAR KIRI (YAW +)
    # --------------------------
    spawn_reset()
    for _ in range(10): env.step(np.zeros(7))
    
    action = np.zeros(7)
    action[0] = -1.0
    action[4] = 1.0 # Yaw + -> Putar Kiri
    for _ in range(30): env.step(action)
    _, end_ang = get_velocity()
    yaw_rate = end_ang[2]
    
    # Putar Kiri = CCW = Rate Positif
    status = "OK" if yaw_rate > 0.05 else "GAGAL"
    logger.info(f"2. Putar Kiri (Yaw+):        {status} | Rate = {yaw_rate:.4f} rad/s")

    # 3. TEST MIRING BELAKANG (PITCH +)
    # ---------------------------------
    # Pitch + -> Nose Up -> Miring Belakang
    spawn_reset()
    for _ in range(10): env.step(np.zeros(7))
    
    action = np.zeros(7)
    action[0] = -1.0
    action[3] = 1.0 # Pitch +
    for _ in range(30): env.step(action)
    _, end_ang = get_velocity()
    pitch_rate = end_ang[1]
    
    # Hidung Naik (Miring Belakang) = Rotasi Y Negatif (Right Hand Rule)
    status = "OK" if pitch_rate < -0.05 else "GAGAL"
    logger.info(f"3. Miring Belakang (Pitch+): {status} | Rate = {pitch_rate:.4f} rad/s")

    # 4. TEST MIRING DEPAN (PITCH -)
    # ------------------------------
    # Pitch - -> Nose Down -> Miring Depan
    spawn_reset()
    for _ in range(10): env.step(np.zeros(7))
    
    action = np.zeros(7)
    action[0] = -1.0
    action[3] = -1.0 # Pitch -
    for _ in range(30): env.step(action)
    _, end_ang = get_velocity()
    pitch_rate = end_ang[1]
    
    # Hidung Turun (Miring Depan) = Rotasi Y Positif
    status = "OK" if pitch_rate > 0.05 else "GAGAL"
    logger.info(f"4. Miring Depan (Pitch-):    {status} | Rate = {pitch_rate:.4f} rad/s")

    # 5. TEST MIRING KANAN (ROLL +)
    # -----------------------------
    # Roll + -> Miring Kanan
    spawn_reset()
    for _ in range(10): env.step(np.zeros(7))
    
    action = np.zeros(7)
    action[0] = -1.0
    action[5] = 1.0 # Roll +
    for _ in range(30): env.step(action)
    _, end_ang = get_velocity()
    roll_rate = end_ang[0]
    
    # Miring Kanan = Rotasi X Positif
    status = "OK" if roll_rate > 0.05 else "GAGAL"
    logger.info(f"5. Miring Kanan (Roll+):     {status} | Rate = {roll_rate:.4f} rad/s")

    # 6. TEST MIRING KIRI (ROLL -)
    # ----------------------------
    # Roll - -> Miring Kiri
    spawn_reset()
    for _ in range(10): env.step(np.zeros(7))
    
    action = np.zeros(7)
    action[0] = -1.0
    action[5] = -1.0 # Roll -
    for _ in range(30): env.step(action)
    _, end_ang = get_velocity()
    roll_rate = end_ang[0]
    
    # Miring Kiri = Rotasi X Negatif
    status = "OK" if roll_rate < -0.05 else "GAGAL"
    logger.info(f"6. Miring Kiri (Roll-):      {status} | Rate = {roll_rate:.4f} rad/s")
    
    logger.info("========================================")
    env.close()

if __name__ == "__main__":
    verify_dynamics()
