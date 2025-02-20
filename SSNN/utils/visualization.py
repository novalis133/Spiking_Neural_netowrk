import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

class ResultsVisualizer:
    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        
    def plot_training_curves(self, model_name, bp_method):
        results_file = self.results_dir / model_name / f"{bp_method}_{model_name}_Adamax_BS_128.csv"
        df = pd.read_csv(results_file)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curve
        sns.lineplot(data=df, x='Epoch', y='Train Loss', ax=ax1)
        ax1.set_title(f'{model_name} - {bp_method} Loss')
        
        # Accuracy curves
        sns.lineplot(data=df, x='Epoch', y='Train Acc', label='Train', ax=ax2)
        sns.lineplot(data=df, x='Epoch', y='Test Acc', label='Test', ax=ax2)
        ax2.set_title(f'{model_name} - {bp_method} Accuracy')
        
        plt.tight_layout()
        save_path = self.results_dir / model_name / f"{bp_method}_training_curves.png"
        plt.savefig(save_path)
        plt.close()

    def compare_models(self, bp_method):
        dfs = []
        for model_dir in self.results_dir.glob("*SNN"):
            results_file = model_dir / f"{bp_method}_{model_dir.name}_Adamax_BS_128.csv"
            if results_file.exists():
                df = pd.read_csv(results_file)
                df['Model'] = model_dir.name
                dfs.append(df)
        
        if not dfs:
            return
            
        combined_df = pd.concat(dfs)
        
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=combined_df, x='Epoch', y='Test Acc', hue='Model')
        plt.title(f'Model Comparison - {bp_method}')
        plt.savefig(self.results_dir / f"model_comparison_{bp_method}.png")
        plt.close()