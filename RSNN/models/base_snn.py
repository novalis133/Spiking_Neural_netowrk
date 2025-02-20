import torch
import torch.nn as nn
import snntorch as snn
from abc import ABC, abstractmethod

class BaseSNN(nn.Module, ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.layers = self._build_layers()
        
    @abstractmethod
    def _build_layers(self):
        pass
        
    def _create_neuron(self, output=False):
        if self.config.neuron_type == 'leaky':
            return snn.Leaky(beta=self.config.beta, init_hidden=True, output=output)
        elif self.config.neuron_type == 'alpha':
            return snn.Alpha(alpha=self.config.alpha, beta=self.config.beta, init_hidden=True, output=output)
        else:
            raise ValueError(f"Unknown neuron type: {self.config.neuron_type}")
    
    @abstractmethod
    def forward(self, x):
        pass