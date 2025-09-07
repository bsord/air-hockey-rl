# This file has been deleted.

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
import torch
import time
import os
import argparse
# Register custom env
import ml_project.envs


class RenderCallback(BaseCallback):
    def __init__(self, render_freq=1):
        super().__init__()
        self.render_freq = render_freq

    def handle_speed_event(self, env):
        import pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

    def _on_step(self) -> bool:
        env = self.training_env.envs[0]
        base_env = env.unwrapped
        # Render every render_freq steps
        if self.n_calls % self.render_freq == 0:
            base_env.render()
            self.handle_speed_event(env)
        return True


# Argument for visibility
parser = argparse.ArgumentParser()
parser.add_argument('--visible', action='store_true', help='Enable rendering for preview')
parser.add_argument('--blue-random', action='store_true', help='Use random actions for blue agent')
args = parser.parse_args()

VISIBLE = args.visible
BLUE_RANDOM = args.blue_random
render_mode = "human" if VISIBLE else None
env = gym.make("AirHockey-v0", render_mode=render_mode, blue_random=BLUE_RANDOM)

model_path = "sac_air_hockey.zip"
if os.path.exists(model_path):
    print("Loading existing model...")
    model = SAC.load("sac_air_hockey", env=env, verbose=1)
else:
    print("Creating new model...")
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

# Adjust render_freq and fps for speed control
callback = RenderCallback(render_freq=1) if VISIBLE else None

model.learn(total_timesteps=10000, callback=callback, log_interval=1)

# Save model
model.save("sac_air_hockey")

env.close()
