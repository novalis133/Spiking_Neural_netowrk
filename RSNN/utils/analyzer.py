import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class ExperimentAnalyzer:
    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        
    def load_experiment_results(self, experiment_name):
        exp_path = self.results_dir / experiment_name
        metrics_files = list(exp_path.glob('*.json'))
        
        results = []
        for f in metrics_files:
            df = pd.read_json(f)
            df['experiment'] = experiment_name
            results.append(df)
            
        return pd.concat(results)
    
    def compare_experiments(self, metric='avg_reward'):
        experiments = [d.name for d in self.results_dir.iterdir() if d.is_dir()]
        results = []
        
        for exp in experiments:
            data = self.load_experiment_results(exp)
            results.append(data)
            
        combined_results = pd.concat(results)
        
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=combined_results, x='episode', y=metric, hue='experiment')
        plt.title(f'Comparison of {metric} across experiments')
        plt.savefig(self.results_dir / f'comparison_{metric}.png')
        plt.close()
        
        return combined_results