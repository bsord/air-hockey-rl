
import pandas as pd
import numpy as np

# Load episode-level rewards
df = pd.read_csv('optuna_episode_rewards.csv')

# Group by trial and calculate mean, std, and Sharpe ratio
grouped = df.groupby('trial_number').agg(
    mean_reward=('reward', 'mean'),
    std_reward=('reward', 'std'),
    batch_size=('batch_size', 'first'),
    buffer_size=('buffer_size', 'first'),
    ent_coef=('ent_coef', 'first'),
    learning_rate=('learning_rate', 'first')
).reset_index()

epsilon = 1e-8
grouped['sharpe'] = grouped['mean_reward'] / (grouped['std_reward'] + epsilon)

# List top 5 trials by Sharpe ratio
top5 = grouped.sort_values('sharpe', ascending=False).head(5)
print('Top 5 trials by Sharpe ratio:')
print(top5[['trial_number', 'mean_reward', 'std_reward', 'sharpe', 'learning_rate', 'ent_coef', 'batch_size', 'buffer_size']])
