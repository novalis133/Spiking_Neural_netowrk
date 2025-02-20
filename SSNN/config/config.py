from dataclasses import dataclass
from typing import Tuple

@dataclass
class NetworkConfig:
    input_size: int = 784
    output_size: int = 10
    hidden_size: int = 1000
    conv1_channels: int = 12
    conv2_channels: int = 64
    kernel_size: int = 5
    beta: float = 0.5
    alpha: float = 0.9
    slope: int = 50
    num_steps: int = 25

@dataclass
class TrainingConfig:
    batch_size: int = 128
    learning_rate: float = 1e-2
    epochs: int = 100
    patience: int = 10
    min_delta: float = 10
    num_workers: int = 14
    subset_size: int = 10
    optimizer_betas: Tuple[float, float] = (0.9, 0.999)
    optimizer_eps: float = 1e-7