from gymnasium.envs.registration import register

register(
    id="AirHockey-v0",
    entry_point="envs.air_hockey_env:AirHockeyEnv",
)
