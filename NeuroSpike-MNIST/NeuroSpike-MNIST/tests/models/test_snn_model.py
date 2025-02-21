import pytest
import torch
from src.models.snn_model import SNNModel

def test_model_forward():
    model = SNNModel(input_size=784, hidden_size=256, output_size=10)
    x = torch.randn(32, 784)
    output = model(x)
    assert output.shape[1] == 32
    assert output.shape[2] == 10