"""
Hyperparameter tuning script for Physics Air Hockey Environment

Provides automated hyperparameter optimization for RL agents.
"""

import os
import sys
import json
import itertools
import argparse
from typing import Dict, Any, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

from train import AirHockeyTrainer


class HyperparameterTuner:
    """
    Hyperparameter tuning manager for Air Hockey RL agents
    """
    
    def __init__(self, base_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the tuner
        
        Args:
            base_config: Base configuration for training
        """
        self.base_config = base_config or self._default_base_config()
        self.results = []
        self.best_config = None
        self.best_score = float('-inf')
    
    def _default_base_config(self) -> Dict[str, Any]:
        """Default base configuration for tuning"""
        return {
            'num_episodes': 500,  # Shorter episodes for tuning
            'max_steps_per_episode': 2000,
            'save_frequency': 100,
            'log_frequency': 50,
            'env_config': {
                'width': 800,
                'height': 400,
                'maximize_window': False
            }
        }
    
    def define_search_space(self) -> Dict[str, List[Any]]:
        """
        Define the hyperparameter search space
        
        Returns:
            Dict mapping parameter names to lists of values to try
        """
        search_space = {
            'reward_type': ['dense', 'sparse'],
            'learning_rate': [0.001, 0.01, 0.1],
            'batch_size': [32, 64, 128],
            'epsilon': [0.1, 0.2, 0.3],
            'gamma': [0.9, 0.95, 0.99],
            # Add more hyperparameters as needed
        }
        return search_space
    
    def grid_search(self, max_combinations: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform grid search over hyperparameter space
        
        Args:
            max_combinations: Maximum number of combinations to try
            
        Returns:
            Best configuration found
        """
        search_space = self.define_search_space()
        
        # Generate all combinations
        param_names = list(search_space.keys())
        param_values = list(search_space.values())
        combinations = list(itertools.product(*param_values))
        
        # Limit combinations if specified
        if max_combinations and len(combinations) > max_combinations:
            import random
            combinations = random.sample(combinations, max_combinations)
        
        print(f"Starting grid search with {len(combinations)} combinations...")
        
        # Evaluate each combination
        for i, combo in enumerate(combinations):
            config = self._create_config(param_names, combo)
            score = self._evaluate_config(config, i + 1, len(combinations))
            
            self.results.append({
                'config': config,
                'score': score,
                'params': dict(zip(param_names, combo))
            })
            
            # Update best configuration
            if score > self.best_score:
                self.best_score = score
                self.best_config = config
                print(f"New best score: {score:.4f}")
        
        self._save_results()
        return self.best_config
    
    def random_search(self, num_trials: int = 50) -> Dict[str, Any]:
        """
        Perform random search over hyperparameter space
        
        Args:
            num_trials: Number of random configurations to try
            
        Returns:
            Best configuration found
        """
        import random
        
        search_space = self.define_search_space()
        print(f"Starting random search with {num_trials} trials...")
        
        for trial in range(num_trials):
            # Sample random configuration
            config = self.base_config.copy()
            params = {}
            
            for param_name, param_values in search_space.items():
                value = random.choice(param_values)
                config[param_name] = value
                params[param_name] = value
            
            score = self._evaluate_config(config, trial + 1, num_trials)
            
            self.results.append({
                'config': config,
                'score': score,
                'params': params
            })
            
            # Update best configuration
            if score > self.best_score:
                self.best_score = score
                self.best_config = config
                print(f"New best score: {score:.4f}")
        
        self._save_results()
        return self.best_config
    
    def _create_config(self, param_names: List[str], param_values: Tuple) -> Dict[str, Any]:
        """Create configuration from parameter names and values"""
        config = self.base_config.copy()
        for name, value in zip(param_names, param_values):
            config[name] = value
        return config
    
    def _evaluate_config(self, config: Dict[str, Any], trial_num: int, total_trials: int) -> float:
        """
        Evaluate a single configuration
        
        Args:
            config: Configuration to evaluate
            trial_num: Current trial number
            total_trials: Total number of trials
            
        Returns:
            Score for this configuration
        """
        print(f"Trial {trial_num}/{total_trials}: Testing config {config}")
        
        try:
            # Create trainer with this configuration
            trainer = AirHockeyTrainer(config)
            trainer.setup_environment()
            trainer.setup_agent("random")  # Use random agent for now
            
            # Run training (shorter for tuning)
            trainer.train()
            
            # Calculate score based on training results
            score = self._calculate_score(trainer)
            
            # Clean up
            trainer.env.close()
            
            print(f"Trial {trial_num} completed with score: {score:.4f}")
            return score
            
        except Exception as e:
            print(f"Trial {trial_num} failed with error: {e}")
            return float('-inf')
    
    def _calculate_score(self, trainer: AirHockeyTrainer) -> float:
        """
        Calculate score from training results
        
        Args:
            trainer: Completed trainer instance
            
        Returns:
            Score for this training run
        """
        if not trainer.episode_rewards:
            return float('-inf')
        
        # Use average reward over last 10% of episodes as score
        last_episodes = max(1, len(trainer.episode_rewards) // 10)
        score = sum(trainer.episode_rewards[-last_episodes:]) / last_episodes
        
        return score
    
    def _save_results(self):
        """Save tuning results to file"""
        results_file = "tuning_results.json"
        
        # Sort results by score
        sorted_results = sorted(self.results, key=lambda x: x['score'], reverse=True)
        
        output = {
            'best_config': self.best_config,
            'best_score': self.best_score,
            'all_results': sorted_results
        }
        
        with open(results_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Results saved to {results_file}")
        print(f"Best score: {self.best_score:.4f}")
        print(f"Best config: {self.best_config}")
    
    def parallel_search(self, search_type: str = "random", num_trials: int = 50, 
                       max_workers: int = 4) -> Dict[str, Any]:
        """
        Perform parallel hyperparameter search
        
        Args:
            search_type: Type of search ("random" or "grid")
            num_trials: Number of trials for random search
            max_workers: Maximum number of parallel workers
            
        Returns:
            Best configuration found
        """
        print(f"Starting parallel {search_type} search with {max_workers} workers...")
        
        if search_type == "grid":
            search_space = self.define_search_space()
            param_names = list(search_space.keys())
            param_values = list(search_space.values())
            combinations = list(itertools.product(*param_values))
            configs = [self._create_config(param_names, combo) for combo in combinations]
        else:  # random
            configs = self._generate_random_configs(num_trials)
        
        # Execute in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_config = {
                executor.submit(self._evaluate_config_wrapper, config, i + 1, len(configs)): config
                for i, config in enumerate(configs)
            }
            
            # Collect results
            for future in as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    score = future.result()
                    self.results.append({
                        'config': config,
                        'score': score,
                        'params': self._extract_params(config)
                    })
                    
                    if score > self.best_score:
                        self.best_score = score
                        self.best_config = config
                        
                except Exception as e:
                    print(f"Configuration failed: {e}")
        
        self._save_results()
        return self.best_config
    
    def _generate_random_configs(self, num_configs: int) -> List[Dict[str, Any]]:
        """Generate random configurations"""
        import random
        
        search_space = self.define_search_space()
        configs = []
        
        for _ in range(num_configs):
            config = self.base_config.copy()
            for param_name, param_values in search_space.items():
                config[param_name] = random.choice(param_values)
            configs.append(config)
        
        return configs
    
    def _extract_params(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract tuned parameters from config"""
        search_space = self.define_search_space()
        return {key: config.get(key) for key in search_space.keys() if key in config}
    
    @staticmethod
    def _evaluate_config_wrapper(config: Dict[str, Any], trial_num: int, total_trials: int) -> float:
        """Wrapper for parallel execution"""
        tuner = HyperparameterTuner()
        return tuner._evaluate_config(config, trial_num, total_trials)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Tune Air Hockey RL Agent Hyperparameters')
    
    parser.add_argument('--search-type', type=str, default='random',
                        choices=['grid', 'random', 'parallel'],
                        help='Type of hyperparameter search')
    parser.add_argument('--trials', type=int, default=50,
                        help='Number of trials for random search')
    parser.add_argument('--max-combinations', type=int, default=None,
                        help='Maximum combinations for grid search')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers')
    parser.add_argument('--episodes', type=int, default=500,
                        help='Number of episodes per trial')
    
    return parser.parse_args()


def main():
    """Main tuning function"""
    args = parse_arguments()
    
    # Base configuration
    base_config = {
        'num_episodes': args.episodes,
        'max_steps_per_episode': 2000,
        'save_frequency': 100,
        'log_frequency': 50,
        'env_config': {
            'width': 800,
            'height': 400,
            'maximize_window': False
        }
    }
    
    # Initialize tuner
    tuner = HyperparameterTuner(base_config)
    
    try:
        if args.search_type == 'grid':
            best_config = tuner.grid_search(args.max_combinations)
        elif args.search_type == 'random':
            best_config = tuner.random_search(args.trials)
        elif args.search_type == 'parallel':
            best_config = tuner.parallel_search('random', args.trials, args.workers)
        
        print("\nTuning completed!")
        print(f"Best configuration: {best_config}")
        print(f"Best score: {tuner.best_score:.4f}")
        
    except KeyboardInterrupt:
        print("\nTuning interrupted by user")
    except Exception as e:
        print(f"Tuning failed with error: {e}")
        raise


if __name__ == "__main__":
    main()