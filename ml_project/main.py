# Entry point for training and evaluating air hockey RL agents

import gymnasium as gym
import ml_project.envs


def visualize_agent():
    from stable_baselines3 import PPO
    env = gym.make("AirHockey-v0", render_mode="human")
    model = PPO.load("ppo_air_hockey")
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        env.render()
    env.close()

if __name__ == "__main__":
    print("Visualizing trained PPO agent...")
    visualize_agent()
