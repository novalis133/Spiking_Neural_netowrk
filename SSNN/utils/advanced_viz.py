import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import pandas as pd
import numpy as np

class AdvancedVisualizer:
    def __init__(self, results_dir='results'):
        self.results_dir = Path(results_dir)
        
    def create_interactive_comparison(self, experiment_name):
        results = []
        exp_dir = self.results_dir / experiment_name
        
        for result_file in exp_dir.glob('*.csv'):
            df = pd.read_csv(result_file)
            df['Model'] = result_file.stem.split('_')[1]
            df['BP Method'] = result_file.stem.split('_')[0]
            results.append(df)
            
        combined_df = pd.concat(results)
        
        fig = go.Figure()
        for model in combined_df['Model'].unique():
            for bp in combined_df['BP Method'].unique():
                data = combined_df[(combined_df['Model'] == model) & 
                                 (combined_df['BP Method'] == bp)]
                fig.add_trace(go.Scatter(
                    x=data['Epoch'],
                    y=data['Test Acc'],
                    name=f"{model}-{bp}",
                    mode='lines+markers'
                ))
                
        fig.update_layout(
            title=f"Experiment: {experiment_name}",
            xaxis_title="Epoch",
            yaxis_title="Test Accuracy (%)"
        )
        
        return fig
    
    def create_heatmap_comparison(self, experiment_name):
        results = self.load_all_results(experiment_name)
        performance_matrix = pd.pivot_table(
            results,
            values='Test Acc',
            index='Model',
            columns='BP Method',
            aggfunc='max'
        )
        
        fig = px.imshow(performance_matrix,
                       labels=dict(x="BP Method", y="Model", color="Max Test Accuracy"),
                       title=f"Performance Comparison - {experiment_name}")
        return fig