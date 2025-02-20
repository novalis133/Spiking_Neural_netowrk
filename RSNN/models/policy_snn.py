import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from .base_snn import BaseSNN

class PolicySNN(BaseSNN):
    def _build_layers(self):
        layers = nn.ModuleList()
        sizes = [self.config.input_size] + list(self.config.hidden_sizes) + [self.config.output_size]
        
        for i in range(len(sizes)-1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            layers.append(self._create_neuron(output=(i == len(sizes)-2)))
            
        return layers
    
    def forward(self, x):
        x = x.to(self.device)
        spikes = []
        mems = []
        
        # Initialize memories
        memories = [layer.init_leaky() if isinstance(layer, snn.Leaky) else None 
                   for layer in self.layers]
        
        # Forward pass
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                x = layer(x)
            else:
                spike, mem = layer(x, memories[i])
                x = spike
                if i == len(self.layers) - 1:  # Last layer
                    spikes.append(spike)
                    mems.append(mem)
                    
        return torch.stack(spikes, dim=0), torch.stack(mems, dim=0)
    
    def act(self, state, temperature=1.0):
        state = torch.from_numpy(state).float().unsqueeze(0)
        action_logits, _ = self.forward(state)
        action_probs = F.softmax(action_logits.squeeze(0) / temperature, dim=-1)
        
        dist = Categorical(action_probs)
        action = dist.sample()
        
        return action.item(), dist.log_prob(action)