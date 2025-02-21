from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ExperimentConfig:
    name: str
    env_name: str
    network_params: Dict[str, Any]
    training_params: Dict[str, Any]
    
EXPERIMENTS = {
    'cartpole_leaky': ExperimentConfig(
        name='cartpole_leaky',
        env_name='CartPole-v1',
        network_params={
            'hidden_sizes': [64, 32],
            'neuron_type': 'leaky',
            'beta': 0.95
        },
        training_params={
            'num_episodes': 1000,
            'learning_rate': 1e-3,
            'temperature': 1.0
        }
    ),
    'cartpole_alpha': ExperimentConfig(
        name='cartpole_alpha',
        env_name='CartPole-v1',
        network_params={
            'hidden_sizes': [64, 32],
            'neuron_type': 'alpha',
            'alpha': 0.9,
            'beta': 0.95
        },
        training_params={
            'num_episodes': 1000,
            'learning_rate': 5e-4,
            'temperature': 1.2
        }
    )
}