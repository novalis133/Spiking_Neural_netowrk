import torch
import numpy as np

class AccuracyMetric:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.correct = 0
        self.total = 0
    
    def update(self, spike_output, target):
        # Sum spikes over time dimension
        spike_sum = spike_output.sum(dim=0)
        pred = spike_sum.argmax(dim=1)
        self.correct += pred.eq(target).sum().item()
        self.total += target.size(0)
    
    def compute(self):
        return self.correct / self.total if self.total > 0 else 0

class SpikeRateMetric:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.total_spikes = 0
        self.total_neurons = 0
        self.total_timesteps = 0
    
    def update(self, spike_output):
        self.total_spikes += spike_output.sum().item()
        self.total_neurons += spike_output.size(1) * spike_output.size(2)
        self.total_timesteps += spike_output.size(0)
    
    def compute(self):
        denominator = self.total_neurons * self.total_timesteps
        return self.total_spikes / denominator if denominator > 0 else 0