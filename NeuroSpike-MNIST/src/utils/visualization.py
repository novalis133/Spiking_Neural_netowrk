import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class SpikeVisualizer:
    def __init__(self):
        try:
            plt.style.use('seaborn')
        except:
            # Fallback to default style if seaborn is not available
            plt.style.use('default')
    
    def plot_spike_train(self, spikes, title="Spike Train", save_path=None):
        """
        Plot spike trains over time
        spikes: tensor of shape (time_steps, batch_size, neurons)
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        spike_matrix = spikes.cpu().numpy()
        
        sns.heatmap(spike_matrix[:, 0, :].T, 
                   cmap='binary',
                   cbar_kws={'label': 'Spike'},
                   ax=ax)
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Neuron Index')
        ax.set_title(title)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_training_progress(self, losses, accuracies, save_path=None):
        """
        Plot training metrics over epochs
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot losses
        ax1.plot(losses, label='Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss Over Time')
        ax1.legend()
        
        # Plot accuracies
        ax2.plot(accuracies, label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Validation Accuracy Over Time')
        ax2.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_membrane_potential(self, membrane_potentials, threshold=None, save_path=None):
        """
        Plot membrane potential dynamics
        membrane_potentials: tensor of shape (time_steps, batch_size, neurons)
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        mem = membrane_potentials.cpu().numpy()
        
        time_steps = np.arange(mem.shape[0])
        ax.plot(time_steps, mem[:, 0, :])
        
        if threshold is not None:
            ax.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Membrane Potential')
        ax.set_title('Membrane Potential Dynamics')
        
        if save_path:
            plt.savefig(save_path)
        plt.show()