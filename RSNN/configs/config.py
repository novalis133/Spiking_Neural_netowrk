from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class NetworkConfig:
    input_size: int = 4
    hidden_sizes: List[int] = (64, 32)
    output_size: int = 2
    beta: float = 0.95
    alpha: float = 0.9
    spike_grad: str = 'atan'
    neuron_type: str = 'leaky'

@dataclass
class RLConfig:
    algorithm: str = 'reinforce'
    gamma: float = 0.99
    temperature: float = 1.0
    entropy_weight: float = 0.01
    value_weight: float = 0.5

@dataclass
class TrainingConfig:
    num_episodes: int = 1000
    max_steps: int = 500
    batch_size: int = 128
    learning_rate: float = 1e-3
    optimizer: str = 'adam'
    checkpoint_freq: int = 100
    eval_freq: int = 50