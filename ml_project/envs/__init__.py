from gymnasium.envs.registration import register

register(
    id="AirHockey-v0",
    entry_point="ml_project.envs.air_hockey_env:AirHockeyEnv",
)
