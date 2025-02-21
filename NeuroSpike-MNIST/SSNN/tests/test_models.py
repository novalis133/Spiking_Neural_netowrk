import unittest
import torch
from SSNN.config.config import NetworkConfig
from SSNN.models.neurons import LeakySNN, AlphaSNN, SynapticSNN, LapicqueSNN

class TestModels(unittest.TestCase):
    def setUp(self):
        self.config = NetworkConfig()
        self.spike_grad = lambda x: x
        self.input = torch.randn(1, 1, 28, 28)
        
    def test_leaky_snn(self):
        model = LeakySNN(self.config, self.spike_grad)
        output = model(self.input)
        self.assertEqual(output.shape[-1], self.config.output_size)
        
    def test_alpha_snn(self):
        model = AlphaSNN(self.config, self.spike_grad)
        output = model(self.input)
        self.assertEqual(output.shape[-1], self.config.output_size)