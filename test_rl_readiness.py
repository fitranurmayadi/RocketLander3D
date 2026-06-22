import gymnasium as gym
import numpy as np
import rocket_lander
import cv2
import os

def test_rl_readiness():
    print("Testing RocketLanderEnv RL Readiness...")
    
    # 1. Test Observation Normalization (Default: True)
    env = gym.make("RocketLander-v0", render_mode=None, normalize_obs=True)
    obs, _ = env.reset()
    print(f"Normalized Reset Obs Shape: {obs.shape}")
    print(f"Normalized Obs Range: [{np.min(obs):.4f}, {np.max(obs):.4f}]")
    
    # Pos/Vel/Alt should be small roughly [-1, 1] even at spawn (alt 50)
    # Alt 50 / 500 = 0.1. Pos 0 / 500 = 0.
    if np.max(np.abs(obs)) > 5.0:
        print("WARNING: Normalized observation values are unexpectedly large!")
    else:
        print("SUCCESS: Observation normalization looks reasonable.")
    env.close()

    # 2. Test Raw Observations (Compatibility Mode)
    env_raw = gym.make("RocketLander-v0", render_mode=None, normalize_obs=False)
    obs_raw, _ = env_raw.reset()
    print(f"Raw Reset Obs Alt (expecting ~50): {obs_raw[17]:.2f}")
    if abs(obs_raw[17] - 50.0) < 5.0:
        print("SUCCESS: Raw observations maintained correctly.")
    else:
        print(f"FAILURE: Raw observation Alt is incorrect: {obs_raw[17]}")
    env_raw.close()

    # 3. Test rgb_array rendering
    print("Testing rgb_array rendering...")
    env_rgb = gym.make("RocketLander-v0", render_mode="rgb_array")
    env_rgb.reset()
    frame = env_rgb.render()
    if frame is not None and isinstance(frame, np.ndarray):
        print(f"SUCCESS: rgb_array render captured frame with shape {frame.shape}")
        # Save a test frame
        cv2.imwrite("rl_render_test.png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print("Saved render test to 'rl_render_test.png'")
    else:
        print("FAILURE: rgb_array render failed or returned None.")
    env_rgb.close()

    # 4. Test Reward Signal
    print("Testing Reward Signal...")
    env_rew = gym.make("RocketLander-v0", render_mode=None)
    obs, _ = env_rew.reset()
    total_reward = 0
    for _ in range(10):
        obs, reward, term, trunc, _ = env_rew.step(env_rew.action_space.sample())
        total_reward += reward
    print(f"Reward after 10 random steps: {total_reward:.4f}")
    if total_reward != 0:
        print("SUCCESS: Reward signal is active.")
    else:
        print("FAILURE: Reward signal is zero.")
    env_rew.close()

if __name__ == "__main__":
    test_rl_readiness()
