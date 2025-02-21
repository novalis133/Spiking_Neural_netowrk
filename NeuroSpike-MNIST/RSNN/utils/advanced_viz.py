import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from IPython.display import clear_output
import time

class AdvancedVisualizer:
    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.is_plotting = False
        
    def plot_neuron_dynamics(self, spikes, membrane_potentials, neuron_indices=None):
        if neuron_indices is None:
            neuron_indices = range(spikes.shape[1])
            
        fig = make_subplots(rows=2, cols=1,
                           subplot_titles=('Spike Activity', 'Membrane Potential'))
        
        # Plot spike activity
        fig.add_trace(
            go.Heatmap(z=spikes[:, neuron_indices].cpu().numpy(),
                      colorscale='Viridis'),
            row=1, col=1
        )
        
        # Plot membrane potentials
        for i in neuron_indices:
            fig.add_trace(
                go.Scatter(y=membrane_potentials[:, i].cpu().numpy(),
                          name=f'Neuron {i}'),
                row=2, col=1
            )
            
        fig.update_layout(height=800, title_text="Neuron Dynamics")
        fig.write_html(self.save_dir / 'neuron_dynamics.html')
        
    def plot_network_activity(self, layer_activations, layer_names):
        num_layers = len(layer_activations)
        fig = plt.figure(figsize=(15, 3*num_layers))
        
        for i, (act, name) in enumerate(zip(layer_activations, layer_names)):
            plt.subplot(num_layers, 1, i+1)
            sns.heatmap(act.cpu().numpy(), cmap='viridis')
            plt.title(f'Layer: {name}')
            plt.xlabel('Time Step')
            plt.ylabel('Neuron Index')
            
        plt.tight_layout()
        plt.savefig(self.save_dir / 'network_activity.png')
        plt.close()
        
    def plot_reward_distribution(self, rewards, window_size=100):
        plt.figure(figsize=(12, 6))
        
        # Plot histogram
        plt.subplot(1, 2, 1)
        sns.histplot(rewards, kde=True)
        plt.title('Reward Distribution')
        plt.xlabel('Reward')
        plt.ylabel('Count')
        
        # Plot rolling statistics
        plt.subplot(1, 2, 2)
        rolling_mean = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        rolling_std = pd.Series(rewards).rolling(window_size).std()
        
        plt.plot(rolling_mean, label='Rolling Mean')
        plt.fill_between(range(len(rolling_std)), 
                        rolling_mean - rolling_std,
                        rolling_mean + rolling_std,
                        alpha=0.2)
        plt.title('Reward Statistics')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'reward_analysis.png')
        plt.close()
        
    def create_training_dashboard(self, metrics_history):
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Learning Curve', 'Policy Loss',
                          'Spike Rates', 'Action Distribution',
                          'Episode Length', 'Temperature Schedule')
        )
        
        # Learning curve
        fig.add_trace(
            go.Scatter(y=metrics_history['rewards'], name='Reward'),
            row=1, col=1
        )
        
        # Policy loss
        fig.add_trace(
            go.Scatter(y=metrics_history['policy_losses'], name='Loss'),
            row=1, col=2
        )
        
        # Spike rates
        fig.add_trace(
            go.Scatter(y=metrics_history['spike_rates'], name='Spike Rate'),
            row=2, col=1
        )
        
        # Action distribution
        fig.add_trace(
            go.Bar(y=metrics_history['action_counts'], name='Actions'),
            row=2, col=2
        )
        
        # Episode length
        fig.add_trace(
            go.Scatter(y=metrics_history['episode_lengths'], name='Length'),
            row=3, col=1
        )
        
        # Temperature
        fig.add_trace(
            go.Scatter(y=metrics_history['temperatures'], name='Temperature'),
            row=3, col=2
        )
        
        fig.update_layout(height=1200, width=1000, title_text="Training Dashboard")
        fig.write_html(self.save_dir / 'training_dashboard.html')
    
    def plot_3d_neuron_activity(self, spikes, membrane_potentials, time_window=100):
        fig = go.Figure()
        
        # Create 3D surface plot
        x = np.arange(spikes.shape[1])  # Neuron indices
        y = np.arange(time_window)      # Time steps
        X, Y = np.meshgrid(x, y)
        
        # Plot spike surface
        fig.add_trace(go.Surface(
            x=X, y=Y, z=spikes[-time_window:].cpu().numpy(),
            colorscale='Viridis',
            name='Spike Activity'
        ))
        
        # Plot membrane potential surface
        fig.add_trace(go.Surface(
            x=X, y=Y, 
            z=membrane_potentials[-time_window:].cpu().numpy(),
            colorscale='Plasma',
            name='Membrane Potential',
            visible='legendonly'
        ))
        
        fig.update_layout(
            scene=dict(
                xaxis_title='Neuron Index',
                yaxis_title='Time Step',
                zaxis_title='Activity'
            ),
            title='3D Neuron Activity Visualization'
        )
        
        fig.write_html(self.save_dir / '3d_neuron_activity.html')
        
    def plot_3d_network_state(self, layer_states, layer_names):
        fig = go.Figure()
        
        for i, (state, name) in enumerate(zip(layer_states, layer_names)):
            state_np = state.cpu().numpy()
            
            fig.add_trace(go.Volume(
                x=np.arange(state_np.shape[0]),
                y=np.arange(state_np.shape[1]),
                z=np.arange(state_np.shape[2]),
                value=state_np,
                opacity=0.1,
                surface_count=20,
                name=f'Layer {name}'
            ))
            
        fig.update_layout(
            scene=dict(
                xaxis_title='Time',
                yaxis_title='Batch',
                zaxis_title='Neuron'
            ),
            title='3D Network State'
        )
        
        fig.write_html(self.save_dir / '3d_network_state.html')
        
    def start_real_time_plotting(self, update_interval=1.0):
        self.is_plotting = True
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        plt.ion()
        
        while self.is_plotting:
            clear_output(wait=True)
            self.fig.show()
            time.sleep(update_interval)
            
    def update_real_time_plot(self, metrics):
        if not self.is_plotting:
            return
            
        for ax in self.axes.flat:
            ax.clear()
            
        # Update reward plot
        self.axes[0,0].plot(metrics['rewards'])
        self.axes[0,0].set_title('Rewards')
        
        # Update loss plot
        self.axes[0,1].plot(metrics['policy_losses'])
        self.axes[0,1].set_title('Policy Loss')
        
        # Update spike rate plot
        self.axes[1,0].plot(metrics['spike_rates'])
        self.axes[1,0].set_title('Spike Rates')
        
        # Update action distribution
        self.axes[1,1].bar(range(len(metrics['action_counts'])), 
                          metrics['action_counts'])
        self.axes[1,1].set_title('Action Distribution')
        
        self.fig.tight_layout()
        
    def stop_real_time_plotting(self):
        self.is_plotting = False
        plt.ioff()
        plt.close(self.fig)