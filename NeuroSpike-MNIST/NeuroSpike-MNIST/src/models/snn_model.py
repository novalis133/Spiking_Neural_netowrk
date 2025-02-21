import torch
import torch.nn as nn
import snntorch as snn

class SNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_steps=100, beta=0.95):
        super().__init__()
        self.num_steps = num_steps
        
        # Initialize layers
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=beta)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.lif2 = snn.Leaky(beta=beta)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.lif3 = snn.Leaky(beta=beta)
        
    def forward(self, x):
        # Initialize hidden states and outputs
        spk1, spk2, spk3 = [], [], []
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        
        # Simulate network for num_steps
        for _ in range(self.num_steps):
            cur1 = self.input_layer(x)
            spk1_out, mem1 = self.lif1(cur1, mem1)
            
            cur2 = self.hidden_layer(spk1_out)
            spk2_out, mem2 = self.lif2(cur2, mem2)
            
            cur3 = self.output_layer(spk2_out)
            spk3_out, mem3 = self.lif3(cur3, mem3)
            
            spk1.append(spk1_out)
            spk2.append(spk2_out)
            spk3.append(spk3_out)
        
        # Stack outputs
        spk1 = torch.stack(spk1, dim=0)
        spk2 = torch.stack(spk2, dim=0)
        spk3 = torch.stack(spk3, dim=0)
        
        return spk3