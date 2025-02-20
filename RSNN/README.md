# RSNN (Reinforcement Spiking Neural Network)

## Overview
A deep reinforcement learning framework using Spiking Neural Networks (SNNs) for continuous control tasks. This implementation combines the efficiency of SNNs with the power of reinforcement learning algorithms.

## Features
- Multiple SNN architectures (Leaky, Alpha, Synaptic)
- Various reinforcement learning algorithms
- Advanced visualization tools
- Real-time training monitoring
- Multi-environment support
- 3D visualization of neural activity

## Project Structure
```plaintext
RSNN/
├── agents/         # RL agents implementation
├── configs/        # Configuration files
├── environments/   # Environment wrappers
├── models/         # Neural network models
├── training/       # Training utilities
├── utils/         # Helper functions
└── results/       # Experiment results
```

## Quick Start
1. Build and run using Docker:
```bash
docker build -t rsnn .
docker run -it --gpus all rsnn
```
2. OR setup locally:
```bash
conda env create -f environment.yml
conda activate rsnn
```
3. Run experiments:
```bash
python main.py
```

## Documentation
- configs/ : Configuration files for experiments
- models/ : SNN implementations
- agents/ : RL agent implementations
- environments/ : Environment wrappers
- utils/ : Visualization and analysis tools

## Results
Results are saved in the results/ directory:

- Training metrics
- Neural activity visualizations
- Performance analysis
- 3D network state visualization

## Development Setup

### Pre-commit Hooks
This project uses pre-commit hooks to maintain code quality. To set up:

1. Install pre-commit:
```bash
conda activate rsnn
pre-commit install
```

