import gymnasium as gym
import numpy as np
import os
import sys
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import rocket_lander

def enjoy(model_path, stats_path, num_episodes=5):
    print(f"Loading model from: {model_path}")
    
    # 1. Create Environment
    # Use render_mode='human' for visualization
    # normalize_obs=False as we use VecNormalize to load stats
    # randomize_spawn=True to see how robust it is
    env = gym.make("RocketLander-v0", render_mode="human", normalize_obs=False, randomize_spawn=True)
    
    # 2. Wrap and Load Normalization Stats
    env = DummyVecEnv([lambda: env])
    if os.path.exists(stats_path):
        env = VecNormalize.load(stats_path, env)
        # Deactivate training mode for normalization
        env.training = False
        env.norm_reward = False
        print("Loaded normalization statistics.")
    else:
        print("WARNING: Normalization statistics pkl not found. Result may be erratic.")

    # 3. Load Policy
    model = PPO.load(model_path, env=env)

    # 4. Run Episodes
    for i in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        print(f"Episode {i+1} starting...")
        while not done:
            # Predict action (deterministic=True is better for evaluation)
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            steps += 1
            env.render()
            
            if done:
                print(f"Episode {i+1} finished. Steps: {steps}, Total Reward: {total_reward:.2f}")
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="./models/ppo_rocket_v2/best_model/best_model.zip")
    parser.add_argument("--stats", type=str, default="./models/ppo_rocket_v2/vec_normalize.pkl")
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()
    
    if os.path.exists(args.model):
        enjoy(args.model, args.stats, args.episodes)
    else:
        print(f"Error: Model not found at {args.model}")
        print("Please run train_ppo_v2.py first.")
