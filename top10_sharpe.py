import pandas as pd
import numpy as np

# Load episode-level rewards
# Change filename if needed
filename = 'optuna_episode_rewards.csv'
df = pd.read_csv(filename)

# Group by trial and calculate mean, std, and Sharpe ratio
results = df.groupby('trial_number').agg(
    mean_reward=('reward', 'mean'),
    std_reward=('reward', 'std'),
    batch_size=('batch_size', 'first'),
    buffer_size=('buffer_size', 'first'),
    ent_coef=('ent_coef', 'first'),
    learning_rate=('learning_rate', 'first')
).reset_index()

# Calculate Sharpe ratio
epsilon = 1e-8
results['sharpe'] = results['mean_reward'] / (results['std_reward'] + epsilon)

# List top 10 trials by Sharpe ratio
top10 = results.sort_values('sharpe', ascending=False).head(10)
print('Top 10 trials by Sharpe ratio:')
print(top10[['trial_number', 'mean_reward', 'std_reward', 'sharpe', 'learning_rate', 'ent_coef', 'batch_size', 'buffer_size']])
