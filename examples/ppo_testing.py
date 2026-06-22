import gymnasium as gym
import numpy as np
import os
import sys
# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rocket_lander

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Lokasi Model
MODEL_PATH = "./models/ppo_rocket_v1/best_model/best_model.zip" # Atau rocket_final.zip
STATS_PATH = "./models/ppo_rocket_v1/vec_normalize.pkl"         # Wajib ada

if __name__ == "__main__":
    # 1. Buat Environment (Single instance untuk render)
    # Gunakan mode HUMAN agar terlihat visualnya
    env = gym.make("RocketLander-v0", render_mode="human")
    env = DummyVecEnv([lambda: env])

    # 2. Load Statistik Normalisasi
    # Kita load env wrapper dari file, lalu kita set env-nya ke env yang baru kita buat
    env = VecNormalize.load(STATS_PATH, env)
    
    # PENTING: Matikan training (update stats) & Matikan normalisasi reward (kita ingin lihat raw score)
    env.training = False
    env.norm_reward = False

    # 3. Load Model AI
    model = PPO.load(MODEL_PATH)

    print("=== MENJALANKAN AI ROCKET LANDER ===")
    
    obs = env.reset()
    while True:
        # Predict action (deterministic=True artinya pakai action terbaik, bukan random)
        action, _states = model.predict(obs, deterministic=True)
        
        obs, rewards, dones, info = env.step(action)
        
        if dones[0]:
            print("Episode Selesai. Resetting...")
            obs = env.reset()