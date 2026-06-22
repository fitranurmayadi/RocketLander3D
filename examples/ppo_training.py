import gymnasium as gym
import numpy as np
import os
import torch
import os
import sys
# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rocket_lander

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env

# --- KONFIGURASI HYPERPARAMETER ---
# PPO Hyperparams yang dioptimalkan untuk Continuous Control (Rocket)
PPO_HYPERPARAMS = {
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,          # Standar yang bagus
    "n_steps": 2048,                # Langkah per update (buffer size)
    "batch_size": 64,               # Ukuran batch update gradient
    "n_epochs": 10,                 # Berapa kali melatih ulang buffer yang sama
    "gamma": 0.99,                  # Discount factor (fokus jangka panjang)
    "gae_lambda": 0.95,             # Faktor bias vs variance
    "clip_range": 0.2,              # PPO Clipping
    "ent_coef": 0.01,               # PENTING: Entropy coefficient agar agen mau eksplorasi di awal
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "policy_kwargs": dict(
        activation_fn=torch.nn.Tanh, # Tanh biasanya lebih baik untuk continuous control
        net_arch=dict(pi=[256, 256], vf=[256, 256]) # Network size (Medium)
    ),
}

# Konfigurasi Training
TOTAL_TIMESTEPS = 2_000_000  # 2 Juta langkah (bisa ditambah nanti)
NUM_ENVS = 8                 # Jumlah simulasi paralel (sesuaikan dengan Core CPU Anda, misal 4, 8, atau 16)
LOG_DIR = "./logs/ppo_rocket_v1/"
MODEL_DIR = "./models/ppo_rocket_v1/"

def make_env(rank, seed=0):
    """
    Utility function for multiprocess env.
    """
    def _init():
        env = gym.make("RocketLander-v0")
        env.reset(seed=seed + rank)
        return env
    return _init
if __name__ == "__main__":
    # 1. Buat Direktori Log
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"Training Start using {NUM_ENVS} parallel environments...")

    # 2. Setup Vectorized Environment (Parallel CPU)
    # Gunakan SubprocVecEnv untuk performa maksimal
    env = SubprocVecEnv([make_env(i) for i in range(NUM_ENVS)])
    
    # 3. Wrap dengan VecMonitor (Log Statistik)
    env = VecMonitor(env, LOG_DIR)

    # 4. Wrap dengan VecNormalize (Wajib untuk PPO!)
    env = VecNormalize(env, 
                       norm_obs=True, 
                       norm_reward=True, 
                       clip_obs=10., 
                       gamma=0.99)

    # =================================================================
    # PERBAIKAN DI SINI: Setup Eval Env yang Benar
    # =================================================================
    # Kita gunakan 'make_vec_env' agar otomatis dibungkus DummyVecEnv dengan bersih
    # Jangan pakai 'lambda: eval_env' yang mereferensi variabel diri sendiri (buggy)
    eval_env = make_vec_env("RocketLander-v0", n_envs=1, seed=12345)
    
    # Wrap Eval Env dengan VecNormalize
    # training=False -> Agar eval env tidak mengupdate statistik mean/variance (hanya pakai punya training)
    # norm_reward=False -> Agar kita melihat Raw Reward (skor asli) saat evaluasi
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False, gamma=0.99)

    # 5. Setup Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000 // NUM_ENVS, 
        save_path=MODEL_DIR, 
        name_prefix="rocket_ppo"
    )
    
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=MODEL_DIR + "best_model",
        log_path=LOG_DIR, 
        eval_freq=50_000 // NUM_ENVS,
        deterministic=True, 
        render=False
    )

    # 6. Inisialisasi Model PPO
    # Tambahkan device='cpu' untuk menghilangkan Warning dan biasanya lebih cepat untuk MLP
    model = PPO(env=env, verbose=1, tensorboard_log=LOG_DIR, device='cpu', **PPO_HYPERPARAMS)

    # 7. Mulai Training
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS, 
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current model...")
    
    # 8. Simpan Model Akhir & Statistik
    model.save(f"{MODEL_DIR}/rocket_final")
    env.save(f"{MODEL_DIR}/vec_normalize.pkl") # PENTING: Simpan statistik normalisasi!
    
    print("Training Selesai.")
    env.close()
    eval_env.close()