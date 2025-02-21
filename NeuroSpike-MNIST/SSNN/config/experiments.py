from dataclasses import dataclass
from typing import List, Dict

@dataclass
class ExperimentConfig:
    name: str
    dataset: str  # 'mnist' or 'fashion_mnist'
    models: List[str]  # list of model names to use
    bp_methods: List[str]  # list of BP methods to use
    surrogate_gradients: List[str]  # list of surrogate gradients to use
    
# Predefined experiments
EXPERIMENTS = {
    'baseline': ExperimentConfig(
        name='baseline',
        dataset='fashion_mnist',
        models=['LeakySNN', 'AlphaSNN', 'SynapticSNN', 'LapicqueSNN'],
        bp_methods=['BPTT', 'RTRL', 'TBPTT'],
        surrogate_gradients=['atan']
    ),
    'surrogate_comparison': ExperimentConfig(
        name='surrogate_comparison',
        dataset='fashion_mnist',
        models=['LeakySNN'],
        bp_methods=['BPTT'],
        surrogate_gradients=['atan', 'sigmoid', 'fast_sigmoid', 'triangular']
    )
}