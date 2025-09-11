import optuna
import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from airhockey.env import MiniAirHockeyEnv

# Evaluation function: runs several episodes and returns mean reward
def evaluate_agent(model, env, n_eval_episodes=30, trial_number=None, params=None):
    rewards = []
    for ep in range(n_eval_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
        # Log each episode's reward to CSV
        if trial_number is not None and params is not None:
            import csv, os
            csv_path = os.path.join(os.getcwd(), 'optuna_episode_rewards.csv')
            write_header = not os.path.exists(csv_path)
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(['trial_number', 'episode', 'reward', 'batch_size', 'buffer_size', 'ent_coef', 'learning_rate'])
                writer.writerow([
                    trial_number,
                    ep,
                    total_reward,
                    params['batch_size'],
                    params['buffer_size'],
                    params['ent_coef'],
                    params['learning_rate']
                ])
    return np.mean(rewards)

# Optuna objective function
def objective(trial):
    # Hyperparameter search space
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    ent_coef = trial.suggest_float('ent_coef', 0.01, 1.0)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    buffer_size = trial.suggest_categorical('buffer_size', [10000, 20000, 50000, 100000])

    env = MiniAirHockeyEnv(paddle_speed=300, center_serve_prob=0.60)
    import logging
    logging.getLogger('stable_baselines3').setLevel(logging.ERROR)
    model = SAC('MlpPolicy', env,
                learning_rate=learning_rate,
                ent_coef=ent_coef,
                batch_size=batch_size,
                buffer_size=buffer_size,
                verbose=0)
    model.learn(total_timesteps=10000)
    params = {
        'batch_size': batch_size,
        'buffer_size': buffer_size,
        'ent_coef': ent_coef,
        'learning_rate': learning_rate
    }
    mean_reward = evaluate_agent(model, env, n_eval_episodes=30, trial_number=trial.number, params=params)
    env.close()
    return mean_reward

if __name__ == '__main__':
    study_name = 'sac_tuning_study'
    study = optuna.create_study(direction='maximize', study_name=study_name)

    def save_trials_csv(study, trial):
        filename = f'{study.study_name}_results.csv'
        study.trials_dataframe().to_csv(filename, index=False)

    study.optimize(objective, n_trials=20, callbacks=[save_trials_csv])
    print('Best hyperparameters:', study.best_params)
    print('Best mean reward:', study.best_value)

    study.trials_dataframe().to_csv(f'{study.study_name}_results.csv', index=False)

    # To run: pip install optuna stable-baselines3 gymnasium
    # Then: python airhockey/tune.py
