"""
Training script for Physics Air Hockey Environment

Provides training functionality for reinforcement learning agents.
"""

import os
import sys
import time
import argparse
from typing import Dict, Any, Optional

from envs.PhysicsAirHockey import PhysicsAirHockeyEnv
from envs.PhysicsAirHockey.rewards import create_reward_calculator


class AirHockeyTrainer:
    """
    Training manager for Air Hockey RL agents
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the trainer
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config or self._default_config()
        self.env = None
        self.agent = None
        self.reward_calculator = None
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_start_time = None
    
    def _default_config(self) -> Dict[str, Any]:
        """Default training configuration"""
        return {
            'num_episodes': 1000,
            'max_steps_per_episode': 5000,
            'reward_type': 'dense',
            'save_frequency': 100,
            'log_frequency': 10,
            'model_save_path': './models/',
            'log_path': './logs/',
            'env_config': {
                'width': 800,
                'height': 400,
                'maximize_window': False
            }
        }
    
    def setup_environment(self):
        """Setup the training environment"""
        print("Setting up environment...")
        env_config = self.config.get('env_config', {})
        self.env = PhysicsAirHockeyEnv(**env_config)
        
        # Setup reward calculator
        reward_type = self.config.get('reward_type', 'dense')
        self.reward_calculator = create_reward_calculator(reward_type)
        
        print(f"Environment created with config: {env_config}")
        print(f"Using {reward_type} reward calculator")
    
    def setup_agent(self, agent_type: str = "random"):
        """
        Setup the RL agent
        
        Args:
            agent_type: Type of agent to create ("random", "dqn", "ppo", etc.)
        """
        print(f"Setting up {agent_type} agent...")
        
        if agent_type == "random":
            self.agent = RandomAgent()
        else:
            # TODO: Implement other agent types (DQN, PPO, SAC, etc.)
            print(f"Agent type '{agent_type}' not implemented yet. Using random agent.")
            self.agent = RandomAgent()
    
    def train(self):
        """Main training loop"""
        if self.env is None or self.agent is None:
            raise ValueError("Environment and agent must be set up before training")
        
        print("Starting training...")
        self.training_start_time = time.time()
        
        num_episodes = self.config['num_episodes']
        max_steps = self.config['max_steps_per_episode']
        
        for episode in range(num_episodes):
            episode_reward, episode_length = self._train_episode(episode, max_steps)
            
            # Log progress
            if episode % self.config['log_frequency'] == 0:
                self._log_progress(episode, episode_reward, episode_length)
            
            # Save model
            if episode % self.config['save_frequency'] == 0:
                self._save_model(episode)
        
        print("Training completed!")
        self._save_final_results()
    
    def _train_episode(self, episode: int, max_steps: int) -> tuple:
        """
        Train for one episode
        
        Args:
            episode: Current episode number
            max_steps: Maximum steps per episode
            
        Returns:
            tuple: (total_reward, episode_length)
        """
        # Reset environment
        self.env.reset_ball()
        self.reward_calculator.reset()
        
        total_reward = 0.0
        step_count = 0
        done = False
        
        while not done and step_count < max_steps:
            # Get current state
            ball_pos = self.env.get_ball_position()
            ball_vel = self.env.get_ball_velocity()
            
            # Agent selects action
            action = self.agent.select_action(ball_pos, ball_vel)
            
            # Apply action (placeholder - implement based on action space)
            self._apply_action(action)
            
            # Step environment
            if not self.env.step():
                done = True
            
            # Calculate reward
            new_ball_pos = self.env.get_ball_position()
            new_ball_vel = self.env.get_ball_velocity()
            reward = self.reward_calculator.calculate_reward(
                new_ball_pos, new_ball_vel, action, done
            )
            
            total_reward += reward
            step_count += 1
            
            # Train agent (if applicable)
            if hasattr(self.agent, 'update'):
                self.agent.update(ball_pos, action, reward, new_ball_pos, done)
        
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(step_count)
        
        return total_reward, step_count
    
    def _apply_action(self, action):
        """
        Apply the selected action to the environment
        
        Args:
            action: Action selected by the agent
        """
        # TODO: Implement action application based on action space definition
        # For now, this is a placeholder
        pass
    
    def _log_progress(self, episode: int, reward: float, length: int):
        """Log training progress"""
        avg_reward = sum(self.episode_rewards[-10:]) / min(10, len(self.episode_rewards))
        avg_length = sum(self.episode_lengths[-10:]) / min(10, len(self.episode_lengths))
        
        elapsed_time = time.time() - self.training_start_time
        
        print(f"Episode {episode:4d} | "
              f"Reward: {reward:8.2f} | "
              f"Avg Reward: {avg_reward:8.2f} | "
              f"Length: {length:4d} | "
              f"Avg Length: {avg_length:6.1f} | "
              f"Time: {elapsed_time:6.1f}s")
    
    def _save_model(self, episode: int):
        """Save the current model"""
        # TODO: Implement model saving
        pass
    
    def _save_final_results(self):
        """Save final training results"""
        # TODO: Implement results saving (metrics, plots, etc.)
        pass


class RandomAgent:
    """Simple random agent for testing"""
    
    def select_action(self, ball_pos, ball_vel):
        """Select a random action"""
        import random
        # TODO: Define proper action space
        return random.choice(['left', 'right', 'up', 'down', 'none'])
    
    def update(self, state, action, reward, next_state, done):
        """Random agent doesn't learn"""
        pass


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Air Hockey RL Agent')
    
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of training episodes')
    parser.add_argument('--agent', type=str, default='random',
                        choices=['random', 'dqn', 'ppo', 'sac'],
                        help='Type of RL agent to use')
    parser.add_argument('--reward', type=str, default='dense',
                        choices=['dense', 'sparse'],
                        help='Type of reward function')
    parser.add_argument('--save-freq', type=int, default=100,
                        help='Model save frequency (episodes)')
    parser.add_argument('--log-freq', type=int, default=10,
                        help='Logging frequency (episodes)')
    
    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_arguments()
    
    # Create training configuration
    config = {
        'num_episodes': args.episodes,
        'reward_type': args.reward,
        'save_frequency': args.save_freq,
        'log_frequency': args.log_freq,
        'max_steps_per_episode': 5000,
        'env_config': {
            'width': 800,
            'height': 400,
            'maximize_window': False
        }
    }
    
    # Initialize trainer
    trainer = AirHockeyTrainer(config)
    trainer.setup_environment()
    trainer.setup_agent(args.agent)
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise
    finally:
        if trainer.env:
            trainer.env.close()


if __name__ == "__main__":
    main()