from gymnasium.envs.registration import register

print("DEBUG: Importing rocket_lander package (Executing __init__.py)")

register(
    id="RocketLander-v0",
    entry_point="rocket_lander.envs:RocketLanderEnv",
    max_episode_steps=60000,
)

print("DEBUG: Registered RocketLander-v0")
