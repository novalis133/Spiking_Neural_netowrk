from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class NetworkConfig:
    input_size: int = 4
    hidden_sizes: List[int] = field(default_factory=lambda: [64, 32])
    output_size: int = 2
    neuron_type: str = 'leaky'
    beta: float = 0.95
    alpha: float = 0.9
    spike_grad: str = 'atan'
    initialization: str = 'kaiming'

@dataclass
class TrainingConfig:
    num_episodes: int = 1000
    max_steps: int = 500
    batch_size: int = 128
    learning_rate: float = 1e-3
    optimizer: str = 'adam'
    scheduler: Optional[Dict[str, Any]] = None
    checkpoint_freq: int = 100
    eval_freq: int = 50
    device: str = 'cuda'

@dataclass
class EnvironmentConfig:
    name: str = 'CartPole-v1'
    max_episode_steps: int = 500
    reward_scale: float = 1.0
    time_limit: Optional[int] = None

@dataclass
class ExperimentConfig:
    name: str = 'default'
    seed: int = 42
    network: NetworkConfig = field(default_factory=NetworkConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    logging_dir: str = 'logs'
    results_dir: str = 'results'