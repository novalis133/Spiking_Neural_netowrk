import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class ExperimentAnalyzer:
    def __init__(self, results_dir='results'):
        self.results_dir = Path(results_dir)
        
    def load_experiment_results(self, experiment_name):
        exp_path = self.results_dir / experiment_name
        results = {}
        
        for result_file in exp_path.glob('*.csv'):
            df = pd.read_csv(result_file)
            model_name = result_file.stem.split('_')[1]
            bp_method = result_file.stem.split('_')[0]
            results[(model_name, bp_method)] = df
            
        return results
    
    def compare_convergence_rates(self, results):
        plt.figure(figsize=(12, 6))
        for (model, bp), df in results.items():
            epochs_to_converge = len(df)
            plt.bar(f"{model}-{bp}", epochs_to_converge)
        plt.xticks(rotation=45)
        plt.title("Convergence Rate Comparison")
        plt.tight_layout()
        
    def analyze_performance(self, results):
        performance_metrics = []
        for (model, bp), df in results.items():
            metrics = {
                'Model': model,
                'BP Method': bp,
                'Max Test Acc': df['Test Acc'].max(),
                'Final Test Acc': df['Test Acc'].iloc[-1],
                'Min Train Loss': df['Train Loss'].min(),
                'Convergence Time': len(df)
            }
            performance_metrics.append(metrics)
        
        return pd.DataFrame(performance_metrics)