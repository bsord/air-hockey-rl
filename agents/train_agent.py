import gymnasium as gym
from stable_baselines3 import PPO
import torch

# Register custom env
import envs

env = gym.make("AirHockey-v0")

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

model.learn(total_timesteps=10000)

# Save model
model.save("ppo_air_hockey")

env.close()
