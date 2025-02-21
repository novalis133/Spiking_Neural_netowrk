import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class Visualizer:
    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_spike_activity(self, spikes, title='Spike Activity'):
        plt.figure(figsize=(12, 6))
        plt.imshow(spikes.cpu().numpy(), aspect='auto', cmap='binary')
        plt.title(title)
        plt.xlabel('Time Step')
        plt.ylabel('Neuron Index')
        plt.colorbar(label='Spike')
        plt.savefig(self.save_dir / f'{title.lower().replace(" ", "_")}.png')
        plt.close()
        
    def plot_training_metrics(self, metrics_dict):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Metrics')
        
        # Plot rewards
        sns.lineplot(data=metrics_dict['rewards'], ax=axes[0,0])
        axes[0,0].set_title('Episode Rewards')
        
        # Plot losses
        sns.lineplot(data=metrics_dict['losses'], ax=axes[0,1])
        axes[0,1].set_title('Policy Losses')
        
        # Plot spike rates
        sns.lineplot(data=metrics_dict['spike_rates'], ax=axes[1,0])
        axes[1,0].set_title('Average Spike Rates')
        
        # Plot temperature
        sns.lineplot(data=metrics_dict['temperatures'], ax=axes[1,1])
        axes[1,1].set_title('Temperature Schedule')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_metrics.png')
        plt.close()