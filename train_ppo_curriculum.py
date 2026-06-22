import gymnasium as gym
import numpy as np
import os
import torch
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.env_util import make_vec_env

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import rocket_lander

class CurriculumCallback(BaseCallback):
    """
    Callback for updating the spawn radius curriculum.
    """
    def __init__(self, total_timesteps, verbose=0):
        super(CurriculumCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        # Save vec_normalize stats periodically so enjoy_rl.py can use them
        if self.n_calls % 10000 == 0:
            stats_path = os.path.join(MODEL_DIR, "vec_normalize.pkl")
            self.training_env.save(stats_path)
            
        progress = self.num_timesteps / self.total_timesteps
        if progress < 0.25:
            radius = 20.0 + (progress / 0.25) * 80.0
        elif progress < 0.5:
            radius = 100.0 + ((progress - 0.25) / 0.25) * 100.0
        else:
            radius = 200.0
        
        # Update all environments in the VecEnv
        self.training_env.env_method("set_spawn_radius", radius)
        
        if self.n_calls % 1000 == 0:
            self.logger.record("train/curr_spawn_radius", radius)
        return True

class CurriculumWrapper(gym.Wrapper):
    """
    Wrapper to handle the dynamic spawn radius.
    """
    def __init__(self, env):
        super().__init__(env)
        self.spawn_radius = 20.0 # Start small
        
    def set_spawn_radius(self, radius):
        self.spawn_radius = radius
        
    def reset(self, **kwargs):
        if "options" not in kwargs or kwargs["options"] is None:
            kwargs["options"] = {}
        kwargs["options"]["spawn_radius"] = self.spawn_radius
        return self.env.reset(**kwargs)

PPO_CONFIG = {
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 128,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "use_sde": True,
    "sde_sample_freq": 4,
    "policy_kwargs": dict(
        activation_fn=torch.nn.Tanh,
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    ),
}

TOTAL_TIMESTEPS = 3_000_000
NUM_ENVS = 8
LOG_DIR = "./logs/ppo_rocket_curriculum/"
MODEL_DIR = "./models/ppo_rocket_curriculum/"

def make_env(rank, seed=0):
    def _init():
        # normalize_obs=False as we use VecNormalize
        env = gym.make("RocketLander-v0", render_mode=None, normalize_obs=False, randomize_spawn=True)
        env = CurriculumWrapper(env)
        env.reset(seed=seed + rank)
        return env
    return _init

if __name__ == "__main__":
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"Starting PPO Curriculum Training with {NUM_ENVS} parallel environments...")

    env = SubprocVecEnv([make_env(i) for i in range(NUM_ENVS)])
    env = VecMonitor(env, LOG_DIR)

    # Eval env (Static fixed difficulty to monitor absolute performance)
    eval_env = make_vec_env("RocketLander-v0", n_envs=1, 
                            env_kwargs={"normalize_obs": False, "randomize_spawn": False})
    eval_env = VecMonitor(eval_env, os.path.join(LOG_DIR, "eval"))
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False, gamma=0.99)

    callbacks = [
        CheckpointCallback(save_freq=max(100_000 // NUM_ENVS, 1), save_path=MODEL_DIR, name_prefix="rocket_ppo"),
        EvalCallback(eval_env, best_model_save_path=os.path.join(MODEL_DIR, "best_model"),
                     log_path=LOG_DIR, eval_freq=max(50_000 // NUM_ENVS, 1), deterministic=True),
        CurriculumCallback(TOTAL_TIMESTEPS)
    ]

    model_path = os.path.join(MODEL_DIR, "rocket_final.zip")
    stats_path = os.path.join(MODEL_DIR, "vec_normalize.pkl")
    
    if os.path.exists(model_path) and os.path.exists(stats_path):
        print(f"Resuming training from {model_path}...")
        # Load existing stats into a wrapper
        env = VecNormalize.load(stats_path, env)
        env.training = True
        env.norm_reward = True
        model = PPO.load(model_path, env=env, device="cpu", verbose=1, tensorboard_log=LOG_DIR)
    else:
        print("Starting new training session...")
        # Create new normalization wrapper
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., gamma=0.99)
        model = PPO(env=env, verbose=1, tensorboard_log=LOG_DIR, device="cpu", **PPO_CONFIG)

    print("Training in progress... Observe 'train/curr_spawn_radius' in Tensorboard.")
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks, progress_bar=True)
    except KeyboardInterrupt:
        print("\nTraining interrupted.")

    model.save(os.path.join(MODEL_DIR, "rocket_final"))
    env.save(os.path.join(MODEL_DIR, "vec_normalize.pkl"))
    print(f"Training complete. Models in {MODEL_DIR}")
