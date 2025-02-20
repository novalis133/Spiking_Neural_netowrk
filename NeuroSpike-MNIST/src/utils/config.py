import yaml
import os
from dataclasses import dataclass

@dataclass
class ModelConfig:
    input_size: int
    hidden_size: int
    output_size: int
    num_steps: int
    beta: float

@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    beta1: float = 0.9
    beta2: float = 0.999

class Config:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        self.model = ModelConfig(**config_dict['model'])
        self.training = TrainingConfig(**config_dict['training'])
        self.data_path = config_dict['data_path']
        
    @staticmethod
    def save_default_config(path):
        default_config = {
            'model': {
                'input_size': 784,
                'hidden_size': 256,
                'output_size': 10,
                'num_steps': 100,
                'beta': 0.95
            },
            'training': {
                'epochs': 100,
                'batch_size': 32,
                'learning_rate': 0.001
            },
            'data_path': 'data/'
        }
        
        with open(path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)