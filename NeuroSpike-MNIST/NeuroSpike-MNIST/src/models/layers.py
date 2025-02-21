import torch
import torch.nn as nn
import snntorch as snn

class SpikingLayer(nn.Module):
    def __init__(self, in_features, out_features, beta=0.95):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.lif = snn.Leaky(beta=beta)
        self.reset_state()
        
    def reset_state(self):
        self.mem = self.lif.init_leaky()
        
    def forward(self, x):
        cur = self.linear(x)
        spk, self.mem = self.lif(cur, self.mem)
        return spk

class SpikingResidualBlock(nn.Module):
    def __init__(self, features, beta=0.95):
        super().__init__()
        self.layer1 = SpikingLayer(features, features, beta)
        self.layer2 = SpikingLayer(features, features, beta)
        
    def forward(self, x):
        identity = x
        out = self.layer1(x)
        out = self.layer2(out)
        return out + identity