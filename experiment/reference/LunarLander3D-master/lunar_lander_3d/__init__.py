"""LunarLander3D - A 3D Lunar Lander environment for reinforcement learning."""

from gymnasium.envs.registration import register

register(
    id="LunarLander3D-v1",
    entry_point="lunar_lander_3d.envs:LunarLander3DEnv",
    max_episode_steps=30000,
)

__version__ = "1.0.0"
