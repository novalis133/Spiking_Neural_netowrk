import numpy as np
from collections import deque

class PerformanceMetrics:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.reset()
        
    def reset(self):
        self.episode_rewards = deque(maxlen=self.window_size)
        self.episode_lengths = deque(maxlen=self.window_size)
        self.policy_losses = deque(maxlen=self.window_size)
        self.spike_rates = deque(maxlen=self.window_size)
        
    def update(self, reward, length, policy_loss, spike_rate):
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.policy_losses.append(policy_loss)
        self.spike_rates.append(spike_rate)
        
    def get_metrics(self):
        return {
            'avg_reward': np.mean(self.episode_rewards),
            'avg_length': np.mean(self.episode_lengths),
            'avg_policy_loss': np.mean(self.policy_losses),
            'avg_spike_rate': np.mean(self.spike_rates),
            'std_reward': np.std(self.episode_rewards),
            'max_reward': np.max(self.episode_rewards),
            'min_reward': np.min(self.episode_rewards)
        }