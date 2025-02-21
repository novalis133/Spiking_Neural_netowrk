from ray import tune
import ray
from functools import partial
import numpy as np

class HyperparameterOptimizer:
    def __init__(self, experiment_config, num_samples=10):
        self.experiment_config = experiment_config
        self.num_samples = num_samples
        
    def objective(self, config):
        modified_config = self.experiment_config.copy()
        modified_config.network_params.update(config['network'])
        modified_config.training_params.update(config['training'])
        
        metrics = run_experiment(modified_config)
        tune.report(mean_reward=metrics['avg_reward'])
        
    def optimize(self):
        search_space = {
            'network': {
                'beta': tune.uniform(0.7, 0.99),
                'hidden_sizes': tune.choice([[32, 16], [64, 32], [128, 64]]),
            },
            'training': {
                'learning_rate': tune.loguniform(1e-4, 1e-2),
                'temperature': tune.uniform(0.5, 2.0)
            }
        }
        
        analysis = tune.run(
            self.objective,
            config=search_space,
            num_samples=self.num_samples,
            metric='mean_reward',
            mode='max'
        )
        
        return analysis.best_config, analysis.best_result