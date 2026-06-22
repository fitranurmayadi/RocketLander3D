import gymnasium as gym
import numpy as np
import os
import torch
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

# Add project root to sys.path to ensure rocket_lander is importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import rocket_lander

# --- PPO HYPERPARAMETERS (Optimized for Continuous Control) ---
PPO_CONFIG = {
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 128,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,           # Encourage exploration
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "use_sde": True,            # State Dependent Exploration (better for smooth control)
    "sde_sample_freq": 4,
    "policy_kwargs": dict(
        activation_fn=torch.nn.Tanh,
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    ),
}

TOTAL_TIMESTEPS = 2_000_000
NUM_ENVS = 8  # Parallel environments
LOG_DIR = "./logs/ppo_rocket_v2/"
MODEL_DIR = "./models/ppo_rocket_v2/"

def make_env(seed=0):
    def _init():
        # normalize_obs=False because we use VecNormalize wrapper
        # randomize_spawn=True for robust training
        env = gym.make("RocketLander-v0", render_mode=None, normalize_obs=False, randomize_spawn=True)
        return env
    return _init

if __name__ == "__main__":
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"Starting PPO training with {NUM_ENVS} parallel environments...")

    # 1. Setup Vectorized Environment
    env = SubprocVecEnv([make_env(i) for i in range(NUM_ENVS)])
    # Monitor should be inside Normalize for stats sync to work correctly in some SB3 versions
    env = VecMonitor(env, LOG_DIR)
    # Add Normalization Wrapper (Crucial for SB3 PPO)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., gamma=0.99)

    # 3. Setup Evaluation Environment
    # We must ensure wrapper structure is identical for sync_envs_normalization
    eval_env = make_vec_env("RocketLander-v0", n_envs=1, 
                            env_kwargs={"normalize_obs": False, "randomize_spawn": False})
    eval_env = VecMonitor(eval_env, os.path.join(LOG_DIR, "eval"))
    # Copy normalization stats from training env during evaluation
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False, gamma=0.99)

    # 4. Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // NUM_ENVS, 1),
        save_path=MODEL_DIR,
        name_prefix="rocket_ppo"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(MODEL_DIR, "best_model"),
        log_path=LOG_DIR,
        eval_freq=max(50_000 // NUM_ENVS, 1),
        deterministic=True,
        render=False
    )

    # 5. Initialize PPO
    model = PPO(env=env, verbose=1, tensorboard_log=LOG_DIR, device="cpu", **PPO_CONFIG)

    # 6. Train
    print("Training in progress... (Check Tensorboard at logs/ppo_rocket_v2/)")
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current progress...")

    # 7. Final Save
    model.save(os.path.join(MODEL_DIR, "rocket_final"))
    env.save(os.path.join(MODEL_DIR, "vec_normalize.pkl"))
    print(f"Training complete. Models saved to {MODEL_DIR}")
