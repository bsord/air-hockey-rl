# Entry point for training and evaluating air hockey RL agents

import gymnasium as gym
import envs

def manual_play():
    env = gym.make("AirHockey-v0", render_mode="human")
    obs, _ = env.reset()
    done = False
    while not done:
        # Example: random actions for both agents
        action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)
        env.render()
    env.close()

if __name__ == "__main__":
    print("Starting Air Hockey RL visualization...")
    manual_play()
